#include "lrc_writer.h"
#include <fstream>
#include <cstdio>
#include <algorithm>

static void format_time_lrc(double ms, char* buf) {
    int total_seconds = (int)(ms / 1000.0);
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;
    int hundredths = (int)((ms / 10.0) - (total_seconds * 100.0 / 10.0)) % 100;
    // More precise calculation
    int ms_int = (int)ms;
    hundredths = (ms_int % 1000) / 10;

    sprintf(buf, "%02d:%02d.%02d", minutes, seconds, hundredths);
}

bool write_lrc(const std::string& output_path,
               const std::vector<AlignedLine>& lines,
               const std::string& title,
               const std::string& artist) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        fprintf(stderr, "[lrc] Failed to open %s for writing\n", output_path.c_str());
        return false;
    }

    // Write metadata
    if (!title.empty()) {
        file << "[ti:" << title << "]\n";
    }
    if (!artist.empty()) {
        file << "[ar:" << artist << "]\n";
    }

    // Sort lines by start time
    std::vector<AlignedLine> sorted_lines = lines;
    std::sort(sorted_lines.begin(), sorted_lines.end(),
              [](const AlignedLine& a, const AlignedLine& b) {
                  return a.start_ms < b.start_ms;
              });

    // Write each line
    for (const auto& line : sorted_lines) {
        char time_buf[32];
        format_time_lrc(line.start_ms, time_buf);
        file << "[" << time_buf << "] " << line.text << "\n";
    }

    file.close();
    fprintf(stderr, "[lrc] Wrote %zu lines to %s\n", lines.size(), output_path.c_str());
    return true;
}
