#include "merger.h"

#include <algorithm>
#include <cmath>
#include <map>

std::vector<AlignedLine> merge_alignment(
    const std::vector<AlignedLine>& ctc_result,
    const std::vector<AlignedLine>& whisper_result)
{
    // Index whisper results by line_index for quick lookup
    std::map<int, const AlignedLine*> whisper_map;
    for (const auto& line : whisper_result) {
        whisper_map[line.line_index] = &line;
    }

    std::vector<AlignedLine> merged;

    // If CTC result is empty, return whisper result
    if (ctc_result.empty()) return whisper_result;
    // If whisper result is empty, return CTC result
    if (whisper_result.empty()) return ctc_result;

    // Use CTC as primary, merge with whisper
    for (const auto& ctc_line : ctc_result) {
        auto it = whisper_map.find(ctc_line.line_index);
        if (it == whisper_map.end()) {
            // No whisper result for this line, use CTC
            merged.push_back(ctc_line);
            continue;
        }

        const AlignedLine& whisper_line = *it->second;

        // If CTC has very low confidence, prefer whisper
        if (ctc_line.confidence < 0.1f && whisper_line.confidence > 0.2f) {
            merged.push_back(whisper_line);
            continue;
        }

        // If whisper has very low confidence, prefer CTC
        if (whisper_line.confidence < 0.1f && ctc_line.confidence > 0.2f) {
            merged.push_back(ctc_line);
            continue;
        }

        // If both have reasonable confidence, average if close, otherwise take CTC
        float conf_diff = std::abs(ctc_line.confidence - whisper_line.confidence);
        double time_diff = std::abs(ctc_line.start_ms - whisper_line.start_ms);

        AlignedLine merged_line = ctc_line;

        if (conf_diff < 0.15f && time_diff < 500.0) {
            // Close enough - average the timestamps
            merged_line.start_ms = (ctc_line.start_ms + whisper_line.start_ms) / 2.0;
            merged_line.end_ms = (ctc_line.end_ms + whisper_line.end_ms) / 2.0;
            merged_line.confidence = std::max(ctc_line.confidence, whisper_line.confidence);
        } else if (ctc_line.confidence < whisper_line.confidence) {
            // Whisper is more confident
            merged_line = whisper_line;
        }
        // else: CTC is more confident (or equal), keep CTC

        merged.push_back(merged_line);
    }

    return merged;
}
