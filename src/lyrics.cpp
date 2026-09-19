#include "lyrics.h"
#include "utils.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

LyricsDocument parse_lyrics(const std::string& path) {
    LyricsDocument doc;

    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "[lyrics] Failed to open %s\n", path.c_str());
        return doc;
    }

    std::string line;
    int line_num = 0;

    // First pass: read all lines and identify chorus blocks
    struct RawLine {
        int num;
        std::string text;
        std::string stripped;
    };
    std::vector<RawLine> raw_lines;

    while (std::getline(file, line)) {
        std::string stripped = utils::trim(line);
        raw_lines.push_back({line_num++, line, stripped});
    }
    file.close();

    // Second pass: identify chorus blocks (lines between "Ref." markers)
    // A chorus block is defined as: lines that follow a "Ref." marker until the next marker
    std::string last_chorus_label;
    std::vector<std::string> current_chorus;
    bool in_chorus = false;
    int chorus_start = -1;

    auto save_chorus = [&]() {
        if (in_chorus && !current_chorus.empty()) {
            doc.choruses[last_chorus_label] = current_chorus;
        }
        current_chorus.clear();
    };

    for (size_t i = 0; i < raw_lines.size(); i++) {
        const auto& rl = raw_lines[i];
        std::string lower = utils::to_lower(rl.stripped);

        // Detect chorus markers: "Ref.", "[Chorus]", "[Chorus N]", etc.
        bool is_marker = false;
        std::string marker_label;

        if (lower == "ref." || lower == "ref" || lower == "ref:") {
            is_marker = true;
            marker_label = "Ref.";
        } else if (lower.substr(0, 7) == "[chorus" || lower == "[ref]" || lower == "[ref.") {
            is_marker = true;
            marker_label = rl.stripped;
        } else if (lower.find("refren") != std::string::npos ||
                   lower.find("refrain") != std::string::npos) {
            is_marker = true;
            marker_label = "Ref.";
        }

        if (is_marker) {
            // Save previous chorus if any
            save_chorus();
            in_chorus = false;
            last_chorus_label = marker_label;
            chorus_start = i + 1;
        } else if (chorus_start >= 0 && !is_marker) {
            // We're after a marker - collect chorus lines
            if (!rl.stripped.empty()) {
                in_chorus = true;
                current_chorus.push_back(rl.stripped);
            }
        }
    }
    save_chorus();

    // Third pass: build final document with expansion
    in_chorus = false;
    std::string current_chorus_label;
    int expanded_index = 0;

    for (size_t i = 0; i < raw_lines.size(); i++) {
        const auto& rl = raw_lines[i];
        std::string lower = utils::to_lower(rl.stripped);

        // Check if this is a marker
        bool is_marker = false;
        std::string marker_label;

        if (lower == "ref." || lower == "ref" || lower == "ref:") {
            is_marker = true;
            marker_label = "Ref.";
        } else if (lower.substr(0, 7) == "[chorus" || lower == "[ref]" || lower == "[ref.") {
            is_marker = true;
            marker_label = rl.stripped;
        } else if (lower.find("refren") != std::string::npos ||
                   lower.find("refrain") != std::string::npos) {
            is_marker = true;
            marker_label = "Ref.";
        }

        if (is_marker) {
            // Add the marker line itself (hidden from alignment but tracked)
            LyricLine ll;
            ll.index = expanded_index++;
            ll.text = rl.stripped;
            ll.normalized = "";
            ll.is_ref = true;
            ll.is_expanded = false;
            doc.lines.push_back(ll);

            // Expand the chorus
            current_chorus_label = marker_label;
            auto it = doc.choruses.find(marker_label);
            if (it != doc.choruses.end()) {
                for (const auto& chorus_line : it->second) {
                    LyricLine cl;
                    cl.index = expanded_index++;
                    cl.text = chorus_line;
                    cl.normalized = utils::normalize(chorus_line);
                    cl.is_ref = false;
                    cl.is_expanded = true;
                    doc.lines.push_back(cl);
                }
            }
        } else if (!rl.stripped.empty()) {
            LyricLine ll;
            ll.index = expanded_index++;
            ll.text = rl.stripped;
            ll.normalized = utils::normalize(rl.stripped);
            ll.is_ref = false;
            ll.is_expanded = false;
            doc.lines.push_back(ll);
        }
    }

    doc.total_lines = doc.lines.size();
    fprintf(stderr, "[lyrics] Parsed %s: %d lines (with expansion)\n", path.c_str(), doc.total_lines);
    fprintf(stderr, "[lyrics] Found %zu chorus blocks\n", doc.choruses.size());

    return doc;
}

std::vector<std::string> get_alignment_text(const LyricsDocument& doc) {
    std::vector<std::string> result;
    for (const auto& line : doc.lines) {
        if (!line.is_ref && !line.normalized.empty()) {
            result.push_back(line.text);
        }
    }
    return result;
}
