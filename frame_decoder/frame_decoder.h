#pragma once

#include <cstdint>

extern "C" {

struct DecodedFrames {
  uint8_t* data;
  int n_frames;
  int height;
  int width;
  int channels;
};

DecodedFrames decode_full_frames(const char* path, float fps_target, int out_width, int out_height);
DecodedFrames decode_minimap_frames(const char* path, float fps_target, float crop_x_pct, float crop_y_pct);
void free_frame_buffer(uint8_t* data);

}
