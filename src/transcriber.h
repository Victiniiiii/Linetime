#pragma once

#include "audio.h"
#include "ctc_aligner.h"

#include <string>
#include <vector>

struct WhisperWord {
    std::string text;
    double start_ms;
    double end_ms;
    float prob;
};

struct WhisperSegment {
    std::string text;
    double start_ms;
    double end_ms;
    std::vector<WhisperWord> words;
};

struct TranscriptionResult {
    std::string language;
    std::vector<WhisperSegment> segments;
    bool success = false;
    std::string error;
};

TranscriptionResult transcribe_audio(const AudioBuffer& audio,
                                     const std::string& model_path,
                                     const std::string& whisper_cli_path,
                                     const std::string& language,
                                     Provider provider,
                                     int threads = 16,
                                     const std::vector<std::string>& environment_additions = {},
                                     const std::string& cache_name = "");

TranscriptionResult load_transcription_json(const std::string& json_path);
