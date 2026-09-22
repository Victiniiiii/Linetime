#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <cstdint>

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
    while (!result.empty() && (result[0] == '#' || result[0] == '*')) result.erase(result.begin());
    result = trim(result);
    while (result.size() >= 2 && result.substr(0, 2) == "**") result.erase(0, 2);
    while (result.size() >= 2 && result.substr(result.size()-2) == "**") result.erase(result.size()-2, 2);
    while (!result.empty() && result[0] == '*') result.erase(result.begin());
    while (!result.empty() && result.back() == '*') result.pop_back();
    return trim(result);
}

// Decode one UTF-8 codepoint; advances i. Returns U+FFFD on invalid.
inline uint32_t utf8_next(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0xFFFD;
    unsigned char c = s[i];
    if (c < 0x80) { i++; return c; }
    if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F);
        i += 2; return cp;
    }
    if ((c >> 4) == 0xE && i + 2 < s.size()) {
        uint32_t cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
        i += 3; return cp;
    }
    if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
        i += 4; return cp;
    }
    i++; return 0xFFFD;
}

// Map a codepoint to an ASCII approximation (may be multi-char). Returns empty if unsupported.
inline std::string map_cp_to_ascii(uint32_t cp) {
    // Latin-1 / Latin Extended common
    switch (cp) {
        case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4: case 0x00C5: return "a";
        case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4: case 0x00E5: return "a";
        case 0x00C7: return "c"; case 0x00E7: return "c";
        case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB: return "e";
        case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB: return "e";
        case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF: return "i";
        case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF: return "i";
        case 0x00D1: return "n"; case 0x00F1: return "n";
        case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6: case 0x00D8: return "o";
        case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6: case 0x00F8: return "o";
        case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC: return "u";
        case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC: return "u";
        case 0x00DD: case 0x0178: return "y"; case 0x00FD: case 0x00FF: return "y";
        case 0x00C6: return "ae"; case 0x00E6: return "ae";
        case 0x00D0: return "d"; case 0x00F0: return "d";
        case 0x00DE: return "th"; case 0x00FE: return "th";
        case 0x00DF: return "ss";
        case 0x0141: return "l"; case 0x0142: return "l";
        case 0x0110: return "d"; case 0x0111: return "d";
        case 0x0152: return "oe"; case 0x0153: return "oe";
        case 0x00A9: case 0x00AE: case 0x00B0: return "";
        default: break;
    }
    // Latin Extended-A
    if (cp >= 0x0100 && cp <= 0x017F) {
        // Even/odd often upper/lower pair sharing base letter for many blocks
        static const char* extA =
            "AaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIiJjJjKkkLlLlLlLlNnNnNnnNnOoOoRrRrRrSsSsSsSsTtTtTtUuUuUuUuUuWwYyYZzZzZz";
        // Manual map for correctness on common Serbian/Turkish/French letters
        switch (cp) {
            case 0x0100: case 0x0101: return "a";
            case 0x0102: case 0x0103: return "a";
            case 0x0104: case 0x0105: return "a";
            case 0x0106: case 0x0107: return "c";
            case 0x0108: case 0x0109: return "c";
            case 0x010A: case 0x010B: return "c";
            case 0x010C: case 0x010D: return "c";
            case 0x010E: case 0x010F: return "d";
            case 0x0110: case 0x0111: return "d";
            case 0x0112: case 0x0113: return "e";
            case 0x0114: case 0x0115: return "e";
            case 0x0116: case 0x0117: return "e";
            case 0x0118: case 0x0119: return "e";
            case 0x011A: case 0x011B: return "e";
            case 0x011C: case 0x011D: return "g";
            case 0x011E: case 0x011F: return "g";
            case 0x0120: case 0x0121: return "g";
            case 0x0122: case 0x0123: return "g";
            case 0x0124: case 0x0125: return "h";
            case 0x0126: case 0x0127: return "h";
            case 0x0128: case 0x0129: return "i";
            case 0x012A: case 0x012B: return "i";
            case 0x012C: case 0x012D: return "i";
            case 0x012E: case 0x012F: return "i";
            case 0x0130: return "i"; case 0x0131: return "i";
            case 0x0132: case 0x0133: return "ij";
            case 0x0134: case 0x0135: return "j";
            case 0x0136: case 0x0137: return "k";
            case 0x0139: case 0x013A: return "l";
            case 0x013B: case 0x013C: return "l";
            case 0x013D: case 0x013E: return "l";
            case 0x013F: case 0x0140: return "l";
            case 0x0141: case 0x0142: return "l";
            case 0x0143: case 0x0144: return "n";
            case 0x0145: case 0x0146: return "n";
            case 0x0147: case 0x0148: return "n";
            case 0x0149: return "n";
            case 0x014A: case 0x014B: return "n";
            case 0x014C: case 0x014D: return "o";
            case 0x014E: case 0x014F: return "o";
            case 0x0150: case 0x0151: return "o";
            case 0x0154: case 0x0155: return "r";
            case 0x0156: case 0x0157: return "r";
            case 0x0158: case 0x0159: return "r";
            case 0x015A: case 0x015B: return "s";
            case 0x015C: case 0x015D: return "s";
            case 0x015E: case 0x015F: return "s";
            case 0x0160: case 0x0161: return "s";
            case 0x0162: case 0x0163: return "t";
            case 0x0164: case 0x0165: return "t";
            case 0x0166: case 0x0167: return "t";
            case 0x0168: case 0x0169: return "u";
            case 0x016A: case 0x016B: return "u";
            case 0x016C: case 0x016D: return "u";
            case 0x016E: case 0x016F: return "u";
            case 0x0170: case 0x0171: return "u";
            case 0x0172: case 0x0173: return "u";
            case 0x0174: case 0x0175: return "w";
            case 0x0176: case 0x0177: return "y";
            case 0x0178: return "y";
            case 0x0179: case 0x017A: return "z";
            case 0x017B: case 0x017C: return "z";
            case 0x017D: case 0x017E: return "z";
            case 0x017F: return "s";
            default: break;
        }
        (void)extA;
    }
    // Turkish extras (Ğğ Şş İı)
    switch (cp) {
        case 0x011E: case 0x011F: return "g";
        case 0x0130: return "i"; case 0x0131: return "i";
        case 0x015E: case 0x015F: return "s";
        default: break;
    }
    // Cyrillic (Russian + Serbian)
    if (cp >= 0x0400 && cp <= 0x04FF) {
        // Build via table for lowercase; uppercase maps to same letter
        uint32_t c = cp;
        // Map uppercase Cyrillic to lowercase range for table lookup
        if (c >= 0x0410 && c <= 0x042F) c = c - 0x0410 + 0x0430; // А-Я -> а-я
        if (c == 0x0401) c = 0x0451; // Ё -> ё
        if (c == 0x0400) c = 0x0450; // Ѐ -> ѐ
        if (c == 0x0402) c = 0x0452; // Ђ -> ђ
        if (c == 0x0403) c = 0x0453; // Ѓ -> ѓ
        if (c == 0x0405) c = 0x0455; // Ѕ -> ѕ
        if (c == 0x0406) c = 0x0456; // І -> і
        if (c == 0x0407) c = 0x0457; // Ї -> ї
        if (c == 0x0408) c = 0x0458; // Ј -> ј
        if (c == 0x0409) c = 0x0459; // Љ -> љ
        if (c == 0x040A) c = 0x045A; // Њ -> њ
        if (c == 0x040B) c = 0x045B; // Ћ -> ћ
        if (c == 0x040C) c = 0x045C; // Ќ -> ќ
        if (c == 0x040E) c = 0x045E; // Ў -> ў
        if (c == 0x040F) c = 0x045F; // Џ -> џ
        switch (c) {
            case 0x0430: return "a";  // а
            case 0x0431: return "b";  // б
            case 0x0432: return "v";  // в
            case 0x0433: return "g";  // г
            case 0x0434: return "d";  // д
            case 0x0435: return "e";  // е
            case 0x0451: return "e";  // ё
            case 0x0436: return "z";  // ж
            case 0x0437: return "z";  // з
            case 0x0438: return "i";  // и
            case 0x0439: return "j";  // й
            case 0x043A: return "k";  // к
            case 0x043B: return "l";  // л
            case 0x043C: return "m";  // м
            case 0x043D: return "n";  // н
            case 0x043E: return "o";  // о
            case 0x043F: return "p";  // п
            case 0x0440: return "r";  // р
            case 0x0441: return "s";  // с
            case 0x0442: return "t";  // т
            case 0x0443: return "u";  // у
            case 0x0444: return "f";  // ф
            case 0x0445: return "h";  // х
            case 0x0446: return "c";  // ц
            case 0x0447: return "c";  // ч
            case 0x0448: return "s";  // ш
            case 0x0449: return "s";  // щ
            case 0x044A: return "";   // ъ (soft sign silent-ish; drop)
            case 0x044B: return "y";  // ы
            case 0x044C: return "";   // ь
            case 0x044D: return "e";  // э
            case 0x044E: return "yu"; // ю
            case 0x044F: return "ya"; // я
            // Serbian
            case 0x0452: return "dj"; // ђ
            case 0x0455: return "z";  // ѕ
            case 0x0456: return "i";  // і
            case 0x0457: return "i";  // ї
            case 0x0458: return "j";  // ј
            case 0x0459: return "lj"; // љ
            case 0x045A: return "nj"; // њ
            case 0x045B: return "c";  // ћ
            case 0x045C: return "c";  // ќ
            case 0x045E: return "u";  // ў
            case 0x045F: return "d";  // џ
            case 0x0400: case 0x0450: return "e";
            default: return "";
        }
    }
    // Greek (basic)
    if (cp >= 0x03B1 && cp <= 0x03C9) {
        static const char* gr = "abgdezhiklmnxoprstufychw";
        // alpha..omega with some gaps - use explicit
        switch (cp) {
            case 0x03B1: return "a"; case 0x03B2: return "b"; case 0x03B3: return "g";
            case 0x03B4: return "d"; case 0x03B5: return "e"; case 0x03B6: return "z";
            case 0x03B7: return "i"; case 0x03B8: return "th"; case 0x03B9: return "i";
            case 0x03BA: return "k"; case 0x03BB: return "l"; case 0x03BC: return "m";
            case 0x03BD: return "n"; case 0x03BE: return "x"; case 0x03BF: return "o";
            case 0x03C0: return "p"; case 0x03C1: return "r"; case 0x03C3: return "s";
            case 0x03C2: return "s"; case 0x03C4: return "t"; case 0x03C5: return "y";
            case 0x03C6: return "f"; case 0x03C7: return "ch"; case 0x03C8: return "ps";
            case 0x03C9: return "o";
            default: (void)gr; return "";
        }
    }
    // CJK / Hangul / other unsupported
    if (cp >= 0x3040 && cp <= 0x9FFF) return ""; // Hiragana/Katakana/CJK/Hangul
    if (cp >= 0xAC00 && cp <= 0xD7AF) return ""; // Hangul syllables
    if (cp >= 0x1100 && cp <= 0x11FF) return ""; // Hangul jamo
    if (cp >= 0x3130 && cp <= 0x318F) return ""; // Hangul compat
    // Arabic, Hebrew, Thai, etc. — unsupported
    if (cp >= 0x0600 && cp <= 0x06FF) return "";
    if (cp >= 0x0900 && cp <= 0x097F) return "";
    if (cp >= 0x0E00 && cp <= 0x0E7F) return "";
    return "";
}

inline std::string normalize(const std::string& s) {
    std::string stripped = strip_markdown(s);
    std::string out;
    out.reserve(stripped.size());
    size_t i = 0;
    while (i < stripped.size()) {
        unsigned char c = stripped[i];
        if (c < 0x80) {
            if (std::isalnum((unsigned char)c) || c == '\'' || c == ' ') {
                out += (char)std::tolower(c);
            } else if (c == '-' || c == '_' || c == '/' || c == '\\' ) {
                out += ' ';
            } else if (c == 0xE2 && i + 2 < stripped.size()) {
                // possible em-dash / smart quote multi-byte — already handled below via utf8
                i++;
            } else {
                // other ASCII punct → drop (or space for separators handled above)
                i++;
                continue;
            }
            i++;
            continue;
        }
        size_t start = i;
        uint32_t cp = utf8_next(stripped, i);
        // Common multi-byte punctuation → space
        if (cp == 0x2014 || cp == 0x2013 || cp == 0x2010 || cp == 0x2011) { out += ' '; continue; } // dashes
        if (cp == 0x2018 || cp == 0x2019 || cp == 0x201B || cp == 0x0060 || cp == 0x00B4) { out += '\''; continue; }
        if (cp == 0x201C || cp == 0x201D || cp == 0x201E || cp == 0x201F) { out += ' '; continue; } // quotes
        if (cp == 0x00A0 || cp == 0x2000 || cp == 0x2001 || cp == 0x2002 || cp == 0x2003) { out += ' '; continue; }
        if (cp == 0x3001 || cp == 0x3002 || cp == 0xFF0C || cp == 0xFF01 || cp == 0xFF1F) { out += ' '; continue; }
        if (cp == 0x00B7 || cp == 0x2022 || cp == 0x2026) { out += ' '; continue; }
        std::string mapped = map_cp_to_ascii(cp);
        if (!mapped.empty()) {
            for (char mc : mapped) out += (char)std::tolower((unsigned char)mc);
        } else if (cp < 0x80) {
            // already handled
        } else {
            // unsupported script — drop codepoint (may yield empty normalized → interpolate)
            (void)start;
        }
    }
    // collapse spaces, tidy apostrophes
    std::string cleaned;
    cleaned.reserve(out.size());
    for (size_t j = 0; j < out.size(); j++) {
        char ch = out[j];
        if (ch == ' ' && (cleaned.empty() || cleaned.back() == ' ')) continue;
        cleaned += ch;
    }
    while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
    // " 'foo" → "foo"; "foo ' " → "foo'"
    if (cleaned.size() >= 2 && cleaned[0] == '\'' && cleaned[1] == ' ') cleaned.erase(0, 2);
    return cleaned;
}

inline std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> words;
    std::istringstream iss(s);
    std::string word;
    while (iss >> word) words.push_back(word);
    return words;
}

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
