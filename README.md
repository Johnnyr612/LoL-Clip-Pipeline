# LoL Clip Pipeline

## Training V2 (improved, anti-overfitting)

### Step 1 - Prepare additional negatives from full clips

Run on local PC (has access to full 60s clips):

```bash
python backend/prepare_negatives.py \
  --clips-dir "C:/Medal/Clips/League of Legends" \
  --clips-dir "D:/Medal/Clips/League of Legends" \
  --labels data/trainer_labels_all.json \
  --output-labels data/trainer_labels_v2.json \
  --precomputed-dir precomputed_v2/ \
  --max-negatives-per-clip 3
```

### Step 2 - Upload to cluster

Upload `precomputed_v2/` folder and `trainer_labels_v2.json`.

### Step 3 - Run v2 training

Use `training-v2` branch on cluster:

```bash
git checkout training-v2
```

Run in notebook:

```python
import sys
from pathlib import Path
from backend import trainer_worker

trainer_worker.PRECOMPUTED_DIR = Path("precomputed_v2")
sys.argv = [
    "trainer_worker.py",
    "--slice", "0",
    "--run-id", "training-v2",
    "--clips-dir", "D:/Medal/Clips/League of Legends",
    "--labels", "data/trainer_labels_v2.json",
    "--output-dir", "checkpoints_v2",
]
raise SystemExit(trainer_worker.main())
```
