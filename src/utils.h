#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace utils {

inline std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::string strip_markdown(const std::string& s) {
    std::string result = s;
    // Strip leading markdown headers
    while (!result.empty() && result[0] == '#') result.erase(result.begin());
    result = trim(result);
    // Strip leading/trailing bold/italic markers
    while (result.size() >= 2 && result.substr(0, 2) == "**") result.erase(0, 2);
    while (result.size() >= 2 && result.substr(result.size()-2) == "**") result.pop_back(), result.pop_back();
    while (!result.empty() && result[0] == '*') result.erase(result.begin());
    while (!result.empty() && result.back() == '*') result.pop_back();
    return trim(result);
}

inline std::string remove_diacritics(const std::string& s) {
    // Simple ASCII approximation for common Latin diacritics
    std::string result;
    result.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = s[i];
        // Check for multi-byte UTF-8 sequences (diacritics are typically 2-byte)
        if (c >= 0xC0 && c < 0xE0 && i + 1 < s.size()) {
            unsigned char c2 = s[i+1];
            // Common Latin Extended characters mapped to ASCII
            switch (c) {
                case 0xC3: // Latin-1 Supplement
                    switch (c2) {
                        case 0x81: result += 'a'; i += 2; continue; // Á
                        case 0x80: result += 'A'; i += 2; continue; // À
                        case 0x86: result += 'C'; i += 2; continue; // Ĉ
                        case 0x89: result += 'E'; i += 2; continue; // É
                        case 0x88: result += 'E'; i += 2; continue; // È
                        case 0x8C: result += 'I'; i += 2; continue; // Ì
                        case 0x8D: result += 'I'; i += 2; continue; // Í
                        case 0x91: result += 'O'; i += 2; continue; // Ò
                        case 0x93: result += 'O'; i += 2; continue; // Ó
                        case 0x96: result += 'O'; i += 2; continue; // Ö
                        case 0x9A: result += 'U'; i += 2; continue; // Ú
                        case 0x9C: result += 'U'; i += 2; continue; // Ü
                        case 0x9E: result += 'Z'; i += 2; continue; // Ž (approx)
                        case 0xA1: result += 'a'; i += 2; continue; // á
                        case 0xA0: result += 'a'; i += 2; continue; // à
                        case 0xA6: result += 'c'; i += 2; continue; // ĉ
                        case 0xA9: result += 'e'; i += 2; continue; // é
                        case 0xA8: result += 'e'; i += 2; continue; // è
                        case 0xAC: result += 'i'; i += 2; continue; // ì
                        case 0xAD: result += 'i'; i += 2; continue; // í
                        case 0xB1: result += 'o'; i += 2; continue; // ò
                        case 0xB3: result += 'o'; i += 2; continue; // ó
                        case 0xB6: result += 'o'; i += 2; continue; // ö
                        case 0xBA: result += 'u'; i += 2; continue; // ú
                        case 0xBC: result += 'u'; i += 2; continue; // ü
                        case 0xBE: result += 'z'; i += 2; continue; // ž (approx)
                        default: break;
                    }
                    break;
                case 0xC4: // Latin Extended-A
                    switch (c2) {
                        case 0x86: result += 'C'; i += 2; continue; // Ć
                        case 0x87: result += 'c'; i += 2; continue; // ć
                        case 0x90: result += 'D'; i += 2; continue; // Ď (approx D)
                        case 0x91: result += 'd'; i += 2; continue; // ď (approx d)
                        case 0x98: result += 'E'; i += 2; continue; // Ę
                        case 0x99: result += 'e'; i += 2; continue; // ę
                        case 0xA0: result += 'L'; i += 2; continue; // Ĺ
                        case 0xA1: result += 'l'; i += 2; continue; // ĺ
                        case 0xA6: result += 'N'; i += 2; continue; // Ń
                        case 0xA7: result += 'n'; i += 2; continue; // ń
                        case 0xB0: result += 'R'; i += 2; continue; // Ŕ
                        case 0xB1: result += 'r'; i += 2; continue; // ŕ
                        case 0xB9: result += 'S'; i += 2; continue; // Š
                        case 0xBA: result += 's'; i += 2; continue; // š
                        case 0xBD: result += 'T'; i += 2; continue; // Ť (approx T)
                        case 0xBE: result += 't'; i += 2; continue; // ť (approx t)
                        case 0xBF: result += 'Z'; i += 2; continue; // Ź
                        default: break;
                    }
                    break;
            }
        } else if (c >= 0xE0 && c < 0xF0 && i + 2 < s.size()) {
            unsigned char c2 = s[i+1], c3 = s[i+2];
            // Serbian Cyrillic in UTF-8 mapped to Latin equivalents
            // These are common Serbian/Bosnian characters
            if (c == 0xD0) {
                switch (c2) {
                    case 0x90: result += 'A'; i += 3; continue; // А
                    case 0x91: result += 'B'; i += 3; continue; // Б
                    case 0x92: result += 'V'; i += 3; continue; // В
                    case 0x93: result += 'G'; i += 3; continue; // Г
                    case 0x94: result += 'D'; i += 3; continue; // Д
                    case 0x95: result += 'Đ'; i += 3; continue; // Ђ -> Đ
                    case 0x96: result += 'E'; i += 3; continue; // Е
                    case 0x97: result += 'Ž'; i += 3; continue; // Ж -> Ž
                    case 0x98: result += 'Z'; i += 3; continue; // З
                    case 0x99: result += 'I'; i += 3; continue; // И
                    case 0x9A: result += 'J'; i += 3; continue; // Ј -> J
                    case 0x9B: result += 'K'; i += 3; continue; // К
                    case 0x9C: result += 'L'; i += 3; continue; // Л
                    case 0x9D: result += 'L'; i += 3; continue; // Љ -> Lj/L
                    case 0x9E: result += 'M'; i += 3; continue; // М
                    case 0x9F: result += 'N'; i += 3; continue; // Н
                    case 0xA0: result += 'N'; i += 3; continue; // Њ -> Nj/N
                    case 0xA1: result += 'O'; i += 3; continue; // О
                    case 0xA2: result += 'P'; i += 3; continue; // П
                    case 0xA3: result += 'R'; i += 3; continue; // Р
                    case 0xA4: result += 'S'; i += 3; continue; // С
                    case 0xA5: result += 'T'; i += 3; continue; // Т
                    case 0xA6: result += 'Ć'; i += 3; continue; // Ћ -> Ć
                    case 0xA7: result += 'U'; i += 3; continue; // У
                    case 0xA8: result += 'F'; i += 3; continue; // Ф
                    case 0xA9: result += 'H'; i += 3; continue; // Х -> H
                    case 0xAA: result += 'C'; i += 3; continue; // Ц -> C
                    case 0xAB: result += 'Č'; i += 3; continue; // Ч -> Č
                    case 0xAC: result += 'D'; i += 3; continue; // Џ -> Dž/D
                    case 0xAD: result += 'Š'; i += 3; continue; // Ш -> Š
                    default: break;
                }
            } else if (c == 0xD1) {
                switch (c2) {
                    case 0x80: result += 'a'; i += 3; continue; // а
                    case 0x81: result += 'b'; i += 3; continue; // б
                    case 0x82: result += 'v'; i += 3; continue; // в
                    case 0x83: result += 'g'; i += 3; continue; // г
                    case 0x84: result += 'd'; i += 3; continue; // д
                    case 0x85: result += 'đ'; i += 3; continue; // ђ -> đ
                    case 0x86: result += 'e'; i += 3; continue; // е
                    case 0x87: result += 'ž'; i += 3; continue; // ж -> ž
                    case 0x88: result += 'z'; i += 3; continue; // з
                    case 0x89: result += 'i'; i += 3; continue; // и
                    case 0x8A: result += 'j'; i += 3; continue; // ј -> j
                    case 0x8B: result += 'k'; i += 3; continue; // к
                    case 0x8C: result += 'l'; i += 3; continue; // л
                    case 0x8D: result += 'l'; i += 3; continue; // љ -> lj/l
                    case 0x8E: result += 'm'; i += 3; continue; // м
                    case 0x8F: result += 'n'; i += 3; continue; // н
                    case 0x90: result += 'n'; i += 3; continue; // њ -> nj/n
                    case 0x91: result += 'o'; i += 3; continue; // о
                    case 0x92: result += 'p'; i += 3; continue; // п
                    case 0x93: result += 'r'; i += 3; continue; // р
                    case 0x94: result += 's'; i += 3; continue; // с
                    case 0x95: result += 't'; i += 3; continue; // т
                    case 0x96: result += 'ć'; i += 3; continue; // ћ -> ć
                    case 0x97: result += 'u'; i += 3; continue; // у
                    case 0x98: result += 'f'; i += 3; continue; // ф
                    case 0x99: result += 'h'; i += 3; continue; // х -> h
                    case 0x9A: result += 'c'; i += 3; continue; // ц -> c
                    case 0x9B: result += 'č'; i += 3; continue; // ч -> č
                    case 0x9C: result += 'd'; i += 3; continue; // џ -> dž/d
                    case 0x9D: result += 'š'; i += 3; continue; // ш -> š
                    default: break;
                }
            }
        }
        result += c;
        ++i;
    }
    return result;
}

inline std::string normalize(const std::string& s) {
    std::string result = strip_markdown(s);
    result = remove_diacritics(result);
    result = to_lower(result);
    // Remove punctuation except apostrophes
    std::string cleaned;
    for (char c : result) {
        if (std::isalnum(c) || c == '\'' || c == ' ') cleaned += c;
    }
    return trim(cleaned);
}

inline std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> words;
    std::istringstream iss(s);
    std::string word;
    while (iss >> word) words.push_back(word);
    return words;
}

// Levenshtein distance for fuzzy matching
inline int levenshtein(const std::string& a, const std::string& b) {
    int m = a.size(), n = b.size();
    std::vector<int> prev(n + 1), curr(n + 1);
    for (int j = 0; j <= n; j++) prev[j] = j;
    for (int i = 1; i <= m; i++) {
        curr[0] = i;
        for (int j = 1; j <= n; j++) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost});
        }
        std::swap(prev, curr);
    }
    return prev[n];
}

inline double similarity(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    int dist = levenshtein(a, b);
    int max_len = std::max(a.size(), b.size());
    return 1.0 - static_cast<double>(dist) / max_len;
}

} // namespace utils
