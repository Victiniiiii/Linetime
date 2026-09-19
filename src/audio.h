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
// Returns empty buffer on failure
AudioBuffer load_audio(const std::string& path);

// Load a 16kHz mono WAV file directly (fast path)
AudioBuffer load_wav_16k_mono(const std::string& path);
