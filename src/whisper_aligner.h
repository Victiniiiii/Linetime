#pragma once

#include "audio.h"
#include "lyrics.h"
#include "ctc_aligner.h"  // for Provider, AlignedLine
#include <string>
#include <vector>

class WhisperAligner {
public:
    WhisperAligner();
    ~WhisperAligner();

    // Initialize with model path and optional provider (CPU/GPU)
    bool init(const std::string& model_path, Provider provider = Provider::Auto);

    // Align audio to lyrics using whisper's DTW token timestamps
    CTCAlignerResult align(const AudioBuffer& audio,
                           const LyricsDocument& lyrics,
                           const std::string& language = "auto");

private:
    struct Impl;
    Impl* impl_;
};
