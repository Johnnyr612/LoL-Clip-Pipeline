"""Local V-JEPA highlight inference, serialized to bound GPU use."""
import contextlib
import math
from pathlib import Path
import threading
import time
from itertools import islice

from . import config

_lock = threading.Lock()
_cached = None


def _load_cached(checkpoint):
    from . import vjepa_runtime as runtime
    import torch
    global _cached
    checkpoint = checkpoint.resolve()
    stat = checkpoint.stat()
    key = (str(checkpoint), stat.st_size, stat.st_mtime_ns, str(config.VJEPA_SOURCE_DIR.resolve()))
    if _cached is None or _cached[0] != key:
        _cached = None
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('architecture') != 'vjepa21_vitb_window_tcn_highlight_v1' or saved.get('schema_version') != 1:
            raise ValueError('Unsupported V-JEPA checkpoint architecture/schema.')
        cfg = runtime.TimelineConfig(**saved['timeline_config'])
        if cfg.image_size != 384 or cfg.spatial_resize != 'full_frame_bilinear':
            raise ValueError('Unsupported V-JEPA preprocessing.')
        if saved['normalization'] != {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}:
            raise ValueError('Unsupported V-JEPA normalization.')
        model = runtime.VJEPAHighlightModel(runtime.construct_encoder(config.VJEPA_SOURCE_DIR), cfg)
        model.load_state_dict(saved['model_state'], strict=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.requires_grad_(False).eval().to(device)
        _cached = (key, model, cfg, saved['decode_defaults'], device)
        del saved
    return _cached


def prepare_model(checkpoint, metrics=None):
    """Load once before shared decoding so sampling uses the saved configuration."""
    started = time.perf_counter()
    with _lock:
        previous = _cached
        loaded = _load_cached(checkpoint)
        if metrics is not None:
            metrics['model_cache_hit'] = previous is loaded
            metrics['model_prepare_seconds'] = round(time.perf_counter()-started, 4)
        return loaded[2]


def predict_trim(source: Path, duration: float, checkpoint: Path, *, decoded=None, metrics=None):
    from .fight_detector import TrimResult
    from . import vjepa_runtime as runtime
    import torch

    if not math.isfinite(duration) or not 0 < duration <= 120:
        raise ValueError('V-JEPA currently supports short clips up to 120 seconds.')
    if not source.is_file():
        raise ValueError(f'Raw video not found: {source}')
    with _lock:
        _load_cached(checkpoint)
        _, model, cfg, decoder, device = _cached
        edges = runtime.timeline_edges(duration, cfg.step_seconds)
        if decoded is not None:
            if decoded.config != cfg or abs(decoded.duration-duration) > 1e-6:
                raise ValueError('Shared frames do not match saved sampling configuration')
            windows = iter(decoded.windows())
        else:
            windows = iter(runtime.sample_windows(source, duration, cfg))
        batch_size = config.VJEPA_WINDOW_BATCH_SIZE if device == 'cuda' else 1
        requested_batch = batch_size
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        parts = []
        retries = 0
        try:
            with torch.inference_mode(), (torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else contextlib.nullcontext()), runtime.tqdm(
                    total=len(edges)-1, desc='V-JEPA highlight windows', unit='window') as progress:
                while True:
                    pending = list(islice(windows, batch_size))
                    if not pending:
                        break
                    offset = 0
                    while offset < len(pending):
                        count = min(batch_size, len(pending)-offset)
                        try:
                            feature_batch = model.encode_windows(torch.stack(pending[offset:offset+count]))
                        except torch.cuda.OutOfMemoryError:
                            if device != 'cuda' or count == 1:
                                raise
                            # Leave the exception scope before clearing allocator cache.
                            batch_size = max(1, count//2)
                            retries += 1
                        else:
                            parts.append(feature_batch.float())
                            offset += count
                            progress.update(count)
                            continue
                        torch.cuda.empty_cache()
                    del pending
                features = torch.cat(parts, dim=0)
                if features.shape != (len(edges)-1, model.encoder.embed_dim):
                    raise ValueError('V-JEPA feature count does not match timeline')
                logits = model.head(features.unsqueeze(0))[0]
            if device == 'cuda':
                torch.cuda.synchronize()
        finally:
            if hasattr(windows, 'close'):
                windows.close()
            if metrics is not None:
                metrics.update({'forward_seconds': round(time.perf_counter()-started, 4),
                                'shared_decode': decoded is not None, 'window_count': len(edges)-1,
                                'requested_batch_size': requested_batch, 'effective_batch_size': batch_size,
                                'oom_retries': retries, 'device': device})
        if not torch.isfinite(logits).all():
            raise ValueError('V-JEPA returned nonfinite outputs.')
        span = runtime.decode_highlight(logits, edges, **decoder)
        if span is None:
            raise ValueError('V-JEPA did not select a highlight above the saved threshold.')
        start, end = span
        # Combat is an auxiliary occupancy head, not the editorial boundary head.
        fight_probs = logits[:, 1].float().sigmoid().cpu()
        bins = [i for i in range(len(fight_probs)) if float(fight_probs[i]) >= 0.5
                and float(edges[i]) < end and float(edges[i+1]) > start]
        fight_start = max(start, float(edges[bins[0]])) if bins else start
        fight_end = min(end, float(edges[bins[-1]+1])) if bins else end
        return TrimResult(clip_start=round(start, 3), clip_end=round(min(end, duration), 3),
                          fight_start=round(fight_start, 3), fight_end=round(fight_end, 3),
                          fight_duration=round(fight_end-fight_start, 3), dialog_segments=[],
                          flags=['highlight_editor_model', 'vjepa21_highlight_editor',
                                 f'highlight_include_peak={logits[:, 0].float().sigmoid().max().item():.3f}'])
