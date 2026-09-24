#pragma once

#include <string>
#include <vector>
#include <optional>

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

// Run whisper-cli and parse JSON output into word-level tokens
TranscriptionResult transcribe_audio(const std::string& audio_path,
                                     const std::string& model_path,
                                     const std::string& language,
                                     int threads = 8);