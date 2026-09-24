#pragma once

#include "lyrics.h"
#include "transcriber.h"
#include <string>
#include <vector>

struct ReconciledLine {
    std::string text;       // final text (hint-corrected or whisper)
    double start_ms;        // from whisper alignment
    double end_ms;          // from whisper alignment
    bool from_hint;         // true if hint text was used
    float confidence;       // alignment confidence 0-1
};

struct ReconcileResult {
    std::vector<ReconciledLine> lines;
    bool success = false;
    std::string error;
};

// Align hint lyrics (LyricsDocument) to whisper transcription (word-level)
// Returns per-line timestamps and corrected text
ReconcileResult reconcile_lyrics(const LyricsDocument& hints,
                                 const TranscriptionResult& whisper,
                                 float similarity_threshold = 0.5f);