#pragma once

#include "ctc_aligner.h"
#include <vector>

// Merge results from two alignment methods
// Takes the best result per line based on confidence, or averages if close
std::vector<AlignedLine> merge_alignment(
    const std::vector<AlignedLine>& ctc_result,
    const std::vector<AlignedLine>& whisper_result);
