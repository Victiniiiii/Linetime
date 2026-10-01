#pragma once

// Whisper tokens that describe the audio rather than the song.
//
// Whisper has to emit *something* for every window it decodes, so silence and
// instrumental passages come back as bracketed stage directions -- [MUSIC],
// [BLINK], [Laughter]. Those are not lyrics and must not reach an LRC.
//
// This lives in a header rather than in reconcile.cpp because both method b
// (audio-only transcription) and method c (transcription plus a lyric hint) need
// the same judgement, and two copies of a list that must stay in step is one
// copy too many. Method b used to skip this entirely and wrote its markers
// straight into the output.
//
// The comparison runs on utils::normalize(), so it is accent- and
// case-insensitive, and Cyrillic is transliterated first: Serbian "Музика"
// arrives here as "muzika", not as the Latin "music". Only the transliterated
// spelling is listed below.
#include "utils.h"
#include "transcriber.h"

namespace whisper_markers {

inline bool is_marker(const std::string& word) {
    static const char* const kMarkers[] = {
        // Stage directions in the original Latin spelling.
        "music", "applause", "laughing", "laughter", "applauding", "singing",
        // Transliterated forms. Whisper picks the script per segment, so the
        // Cyrillic spellings reach the output as real lines on Serbian tracks.
        "muzika", "muziku", "muzicki",   // Serbian "music"
        "aplauz",                        // "applause"
        "smijeh", "smijanje",            // "laughter"
        "pjevanje", "pjeva",             // "singing"
        "ples",                          // "dancing"
        // Subtitling narration. Whisper emits this over a fade-out, naming the
        // subtitle track it believes it is transcribing.
        "subtitrating", "subtitrujuce", "subtitluje",
        nullptr
    };
    const std::string n = utils::normalize(word);
    if (n.empty()) return false;
    for (int i = 0; kMarkers[i]; i++) {
        if (n == kMarkers[i]) return true;
    }
    return false;
}

// True when the segment carries no lyric words at all: every recognised word is
// either a marker or an interjection. A segment that mixes a marker with real
// lyric words is not in this category -- dropping it would lose content.
inline bool segment_is_non_lyric(const WhisperSegment& seg) {
    bool any_word = false;
    for (const auto& w : seg.words) {
        any_word = true;
        if (!is_marker(w.text)) return false;
    }
    // No word timings at all still counts as non-lyric: whisper sometimes reports
    // an empty word list for a silent window, which is the same case.
    (void)any_word;
    return true;
}

}  // namespace whisper_markers
