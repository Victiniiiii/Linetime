#pragma once

#include "audio.h"
#include "lyrics.h"
#include "ctc_aligner.h"  // for AlignedLine
#include <string>
#include <vector>

class WhisperAligner {
public:
    WhisperAligner();
    ~WhisperAligner();

    bool init(const std::string& model_path);

    // Align audio to lyrics using whisper's DTW token timestamps
    CTCAlignerResult align(const AudioBuffer& audio,
                           const LyricsDocument& lyrics,
                           const std::string& language = "auto");

private:
    struct Impl;
    Impl* impl_;
};
