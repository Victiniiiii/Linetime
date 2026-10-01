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

// Spoken broadcast narration that whisper invents over silence or an outro.
//
// This is a separate class from the bracketed markers above: those are stage
// directions whisper *labels* the audio with, while this is a sentence about
// the video. Whisper has to decode something for every 30 s window, and when a
// track ends or fades out over nothing it reaches for the credit line it has
// heard on countless other recordings. Measured on the 8-song test set it
// produced "Hvala sto pratite kanal." on qfln (00:00), 5e1c (03:18) and
// "Hvala na svijanju!" on 0iac (04:07) -- none of which is sung.
//
// Why a phrase list rather than a confidence threshold: every confidence signal
// tried here fails on this text, because whisper is *certain* about a formulaic
// line it has memorised. Measured medians over the same runs were 0.936 for
// real sung segments and 0.846 for fabricated ones -- overlapping ranges, and
// the credit lines themselves scored 0.761 and 0.857. CTC forced-alignment
// confidence, which is computed against the audio rather than whisper's belief,
// did not separate them either (real median 0.605, fabricated 0.585); it only
// caught one of the three.
//
// A phrase list is only safe because these phrases are a closed class. People
// do not sing "thanks for watching". The false-positive rate is the number
// that matters and it was measured, not assumed:
//
//   0 hits in the 270 ground-truth lines of the 8-song test set
//   1 hit in all 2431 ground-truth lines in the database -- and that one,
//     g1um at 03:29.90, is itself the credit line, i.e. the same hallucination
//     already baked into the reference.
//
// The list is deliberately phrases. An earlier version split them into loose
// words and matched 21.9% of real lyric lines, because "na", "je" and "sto" are
// ordinary words in Bosnian. Only whole phrases are matched here.
//
// Every entry must also survive its own utils::normalize(); two did not
// ("svidjanju" for eight letters, "dont" versus "don't") and both bugs were
// caught by re-reading the tool's output rather than the list.
// tools/narration_probe.cpp asserts all of this in milliseconds, with no GPU,
// model or audio involved. Build and run it after editing the list below.
inline bool is_narration(const std::string& text) {
    static const char* const kNarration[] = {
        // Bosnian / Serbian / Croatian. Normalization transliterates Cyrillic,
        // so a Cyrillic rendering of any of these reaches here as the Latin
        // spelling below.
        "hvala sto pratite kanal",
        // "hvala na sviđanju!" -- the spelling matters and has to be read off
        // utils::normalize, not guessed. U+0111 (đ) maps to "d", so s-v-i-đ-a-n-j-u
        // comes out as "svidanju" -- eight letters, one j. Writing "svij" or
        // "svidjanju" makes the filter miss a fabrication it is supposed to
        // catch, which is exactly what happened on 0iac before this was pinned
        // down with a probe.
        "hvala na svidanju",
        "hvala sto poslusate",
        "hvala na poslusanju",
        "hvala na pratnji",
        "hvala vam na pazi",
        "hvala sto gledate",
        "hvala sto dijelite",
        "molim vas da pretplatite",
        "pretplatite se na kanal",
        // English, and the credit roll it appears in.
        "thanks for watching",
        "thanks for listening",
        "thanks for watching this",
        "see you next time",
        // The apostrophe is part of the normalized form: normalize() keeps
        // "'" and maps U+2019 to it, so this must be written "don't", not
        // "dont". Caught by nt.cpp, the probe that asserts every listed phrase
        // survives its own normalization.
        "don't forget to",
        "for more videos",
        "all rights reserved",
        "credits titles",
        // Subscribe / call-to-action, any language it is phrased in.
        "subscribe to my",
        "kliknite na dugme",
        "pritisnite dugme",
        "ukljucite zvuk",
        "stavite na tacne",
        "pogledajte opis",
        "u opisu videa",
        "nastavak slijedi",
        "kraj emisije",
        nullptr
    };
    // Pad with spaces so a phrase can only match on whole words. Without this,
    // "hvala sto poslusate" would also fire on a lyric containing "poslusate".
    const std::string hay = " " + utils::normalize(text) + " ";
    for (int i = 0; kNarration[i]; i++) {
        if (hay.find(std::string(" ") + kNarration[i] + " ") != std::string::npos)
            return true;
    }
    return false;
}

// True when the segment carries no lyric words at all: every recognised word is
// either a marker or an interjection. A segment that mixes a marker with real
// lyric words is not in this category -- dropping it would lose content.
inline bool segment_is_non_lyric(const WhisperSegment& seg) {
    // Checked first: a credit line is built entirely from ordinary words, so the
    // per-word marker test below would pass it through as lyric content.
    if (is_narration(seg.text)) return true;
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
