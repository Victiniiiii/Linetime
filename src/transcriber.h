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
    // whisper's own account of what backends it loaded, e.g.
    // "WHISPER : ... | CUDA : ARCHS = 610 | ...". Read from the output JSON so a
    // GPU request can be verified rather than assumed.
    std::string systeminfo;
    // True only when whisper's own CUDA backend reported finding a device. Set from
    // the child's stderr, not from the flags it was given, because a CPU build
    // accepts the same flags and still runs on the CPU.
    bool gpu_confirmed = false;
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
                                     const std::string& cache_name = "",
                                     float no_speech_threshold = -1.0f);

TranscriptionResult load_transcription_json(const std::string& json_path);
