#pragma once

#include <string>
#include <vector>
#include <map>

struct LyricLine {
    int index;           // original line number (0-based)
    std::string text;    // original text
    std::string normalized; // normalized for matching
    bool is_ref;         // is this a Ref./Chorus marker
    bool is_expanded;    // was this expanded from a Ref. marker
};

struct LyricsDocument {
    std::vector<LyricLine> lines;
    std::map<std::string, std::vector<std::string>> choruses; // marker label -> chorus lines
    int total_lines = 0;
};

// Parse lyrics file, expanding Ref./Chorus markers
// Use "-" as path to read from stdin
LyricsDocument parse_lyrics(const std::string& path);

// Get the plain text of all non-marker lines (for forced alignment input)
std::vector<std::string> get_alignment_text(const LyricsDocument& doc);
