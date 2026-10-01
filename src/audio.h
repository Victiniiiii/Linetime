#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct AudioBuffer {
    std::vector<float> samples;  // mono, 16kHz, float32
    int sample_rate = 16000;
    int n_samples = 0;
    double duration_sec = 0.0;
};

// Load any audio file and convert to 16kHz mono float32
// ffmpeg_path: path to ffmpeg binary (empty = search PATH)
// separate_vocals: keep only the centre channel, where a lead vocal usually sits.
//   This is a mid/side subtraction, not a learned separator, so it is cheap and
//   needs no model, but it also removes whatever else is centred and does nothing
//   for a mono file. It helps segmentation, not transcription accuracy.
AudioBuffer load_audio(const std::string& path, const std::string& ffmpeg_path = "",
                       bool separate_vocals = false);

// Load a 16kHz mono WAV file directly (fast path)
AudioBuffer load_wav_16k_mono(const std::string& path, const std::string& ffmpeg_path = "",
                              bool separate_vocals = false);
