#include "frame_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace {

DecodedFrames empty_result() { return DecodedFrames{nullptr, 0, 0, 0, 3}; }

DecodedFrames decode_frames(const char* path, float fps_target, int out_width, int out_height, bool minimap, float crop_x_pct, float crop_y_pct) {
  AVFormatContext* fmt = nullptr;
  if (avformat_open_input(&fmt, path, nullptr, nullptr) < 0) return empty_result();
  if (avformat_find_stream_info(fmt, nullptr) < 0) {
    avformat_close_input(&fmt);
    return empty_result();
  }

  int video_stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (video_stream < 0) {
    avformat_close_input(&fmt);
    return empty_result();
  }

  AVStream* stream = fmt->streams[video_stream];
  const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  avcodec_parameters_to_context(ctx, stream->codecpar);
  if (avcodec_open2(ctx, codec, nullptr) < 0) {
    avcodec_free_context(&ctx);
    avformat_close_input(&fmt);
    return empty_result();
  }

  double source_fps = av_q2d(stream->avg_frame_rate);
  if (source_fps <= 0) source_fps = 120.0;
  int step = std::max(1, static_cast<int>(std::round(source_fps / fps_target)));
  int native_w = ctx->width;
  int native_h = ctx->height;
  int crop_x = minimap ? static_cast<int>(native_w * crop_x_pct) : 0;
  int crop_y = minimap ? static_cast<int>(native_h * crop_y_pct) : 0;
  int src_w = minimap ? native_w - crop_x : native_w;
  int src_h = minimap ? native_h - crop_y : native_h;
  int dst_w = minimap ? src_w : out_width;
  int dst_h = minimap ? src_h : out_height;

  SwsContext* sws = sws_getContext(src_w, src_h, ctx->pix_fmt, dst_w, dst_h, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
  AVPacket* packet = av_packet_alloc();
  AVFrame* frame = av_frame_alloc();
  std::vector<uint8_t> output;
  int64_t frame_index = 0;

  while (av_read_frame(fmt, packet) >= 0) {
    if (packet->stream_index == video_stream && avcodec_send_packet(ctx, packet) == 0) {
      while (avcodec_receive_frame(ctx, frame) == 0) {
        if (frame_index % step == 0) {
          std::vector<uint8_t> rgb(dst_w * dst_h * 3);
          uint8_t* src_data[4] = {frame->data[0], frame->data[1], frame->data[2], frame->data[3]};
          int src_linesize[4] = {frame->linesize[0], frame->linesize[1], frame->linesize[2], frame->linesize[3]};
          if (minimap && frame->data[0]) {
            // For common packed formats this shifts the crop during conversion. Planar
            // formats are still handled by swscale from the full frame if pointer math
            // cannot be represented safely for every plane.
            src_data[0] = frame->data[0] + crop_y * frame->linesize[0] + crop_x;
          }
          uint8_t* dst_data[4] = {rgb.data(), nullptr, nullptr, nullptr};
          int dst_linesize[4] = {dst_w * 3, 0, 0, 0};
          sws_scale(sws, src_data, src_linesize, 0, src_h, dst_data, dst_linesize);
          output.insert(output.end(), rgb.begin(), rgb.end());
        }
        frame_index++;
      }
    }
    av_packet_unref(packet);
  }

  int n_frames = static_cast<int>(output.size() / (dst_w * dst_h * 3));
  uint8_t* data = nullptr;
  if (!output.empty()) {
    data = static_cast<uint8_t*>(av_malloc(output.size()));
    std::memcpy(data, output.data(), output.size());
  }

  sws_freeContext(sws);
  av_frame_free(&frame);
  av_packet_free(&packet);
  avcodec_free_context(&ctx);
  avformat_close_input(&fmt);
  return DecodedFrames{data, n_frames, dst_h, dst_w, 3};
}

}  // namespace

extern "C" DecodedFrames decode_full_frames(const char* path, float fps_target, int out_width, int out_height) {
  return decode_frames(path, fps_target, out_width, out_height, false, 0.0f, 0.0f);
}

extern "C" DecodedFrames decode_minimap_frames(const char* path, float fps_target, float crop_x_pct, float crop_y_pct) {
  return decode_frames(path, fps_target, 0, 0, true, crop_x_pct, crop_y_pct);
}

extern "C" void free_frame_buffer(uint8_t* data) {
  if (data != nullptr) av_free(data);
}
