#pragma once

#include "audio.h"
#include "lyrics.h"
#include <string>
#include <vector>
#include <map>

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

    // Initialize with model paths
    bool init(const std::string& onnx_model_path,
              const std::string& tokenizer_path);

    // Align audio to lyrics, returning per-line timestamps
    CTCAlignerResult align(const AudioBuffer& audio,
                           const LyricsDocument& lyrics);

private:
    struct Impl;
    Impl* impl_;
};
