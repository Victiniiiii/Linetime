#pragma once

#include "lyrics.h"
#include "transcriber.h"
#include <string>
#include <vector>

struct ReconciledLine {
    std::string text;       // final text (hint-corrected or whisper)
    std::string align_text; // text used for CTC forced alignment (hint spelling
                            // even when `text` was rebuilt from whisper), may be empty
    double start_ms;        // from whisper alignment
    double end_ms;          // from whisper alignment
    bool from_hint;         // true if hint text was used
    bool recovered;         // true if emitted from audio (missing-section fill)
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
                                 float similarity_threshold = 0.5f,
                                 bool recover_missing = false);