#pragma once

#include "ctc_aligner.h"
#include <string>
#include <vector>

// Write alignment results to LRC format
bool write_lrc(const std::string& output_path,
               const std::vector<AlignedLine>& lines,
               const std::string& title = "",
               const std::string& artist = "");
