#pragma once

#include "audio.h"
#include "lyrics.h"
#include <string>
#include <vector>
#include <map>

enum class Provider {
    Auto,    // Detect best available (CUDA > CoreML > CPU)
    CPU,     // CPU only
    CUDA,    // NVIDIA GPU (Linux/Windows)
    CoreML   // Apple GPU/ANE (macOS)
};

struct AlignedWord {
    std::string text;
    int start_frame;
    int end_frame;
    double start_ms;
    double end_ms;
    float confidence;
};

struct AlignedLine {
    int line_index;
    double start_ms;
    double end_ms;
    float confidence;
    std::string text;
    std::string align_text; // hint spelling used for CTC alignment (may be empty)
    std::vector<AlignedWord> words;
};

struct CTCAlignerResult {
    std::vector<AlignedLine> lines;
    bool success = false;
    std::string error;
};

class CTCAligner {
public:
    CTCAligner();
    ~CTCAligner();

    // Initialize with model paths and optional provider selection
    bool init(const std::string& onnx_model_path,
              const std::string& tokenizer_path,
              Provider provider = Provider::Auto);

    // Align audio to lyrics, returning per-line timestamps.
    // boost: non-blank logit boost (production default 5.0)
    CTCAlignerResult align(const AudioBuffer& audio,
                           const LyricsDocument& lyrics,
                           float boost = 5.0f);

    // True when init failed only because an explicitly requested provider was
    // not available, as opposed to a model or tokenizer that could not be
    // loaded. Callers use this to tell "fall back to CPU" apart from "the user
    // asked for CUDA and there is no CUDA", which must not be silent.
    bool provider_unavailable() const;

private:
    struct Impl;
    Impl* impl_;
};
