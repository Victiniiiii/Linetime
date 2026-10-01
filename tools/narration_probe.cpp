// Self-test for the narration phrase list in src/whisper_markers.h.
//
// This exists because the list is only correct if every phrase survives its own
// utils::normalize(), and two entries silently did not:
//
//   "hvala na svidjanju" -- s-v-i-đ-a-n-j-u normalizes to "svidanju" (eight
//     letters, one j). Written with two j's the filter missed a fabrication on
//     0iac that it was supposed to catch.
//   "dont forget to" -- normalize() keeps the apostrophe and maps U+2019 to it,
//     so the normalized form is "don't forget to".
//
// Both were found by re-reading the tool's actual output, not by reading the
// list, which is exactly the failure mode this file makes impossible.
//
// Build and run (no GPU, no model, no audio -- milliseconds):
//
//   g++ -std=c++17 -O0 tools/narration_probe.cpp -o /tmp/narration_probe \
//       -Isrc -pthread && /tmp/narration_probe
//
// Exits non-zero on any failure, so it is usable as a pre-commit check.
#include "whisper_markers.h"

#include <cstdio>

namespace {

int failures = 0;

void must_catch(const char* s) {
    if (!whisper_markers::is_narration(s)) {
        std::printf("  MISSED  \"%s\"  (normalizes to \"%s\")\n", s,
                    utils::normalize(s).c_str());
        failures++;
    }
}

void must_keep(const char* s) {
    if (whisper_markers::is_narration(s)) {
        std::printf("  FALSE POSITIVE  \"%s\"\n", s);
        failures++;
    }
}

}  // namespace

int main() {
    std::printf("narration phrases that must be caught:\n");
    must_catch("Hvala što pratite kanal.");
    must_catch("hvala sto pratite kanal");
    must_catch("Hvala na sviđanju!");
    must_catch("Hvala na svidanju!");
    must_catch("Hvala na poslušanju");
    must_catch("Hvala na pratnji");
    must_catch("Hvala vam na paži");
    must_catch("Hvala što gledate");
    must_catch("Hvala što dijelite");
    must_catch("Molim vas da pretplatite");
    must_catch("Pretplatite se na kanal");
    must_catch("Thanks for watching");
    must_catch("Thanks for listening");
    must_catch("See you next time");
    must_catch("Don't forget to");
    must_catch("For more videos");
    must_catch("All rights reserved");
    must_catch("Credits titles");
    must_catch("Subscribe to my channel");
    must_catch("Kliknite na dugme");
    must_catch("Pritisnite dugme");
    must_catch("Uključite zvuk");
    must_catch("Stavite na tačne");
    must_catch("Pogledajte opis");
    must_catch("U opisu videa");
    must_catch("Nastavak slijedi");
    must_catch("Kraj emisije");

    // Real lyric lines, taken from the ground truth of the 8-song test set.
    // These are the false positives the phrase list must never produce: the
    // measured rate over all 2431 synced lines in the database was 1 hit, and
    // that one was itself a credit line already baked into the reference.
    std::printf("real lyric lines that must NOT be caught:\n");
    must_keep("Na verandi, u lavandi suzu pustila");
    must_keep("Zrelo grožđe brzo prođe, al' je najslađe");
    must_keep("Da je nekad ovo mirno more bilo nemirno");
    must_keep("Ovo je vrijeme kad se uči");
    must_keep("Artiljerija, Bosanac sam bekrija");
    must_keep("Modra boja na bijelome platnu");
    must_keep("Na krugima svi nosimo zlatne ljiljane");
    must_keep("Sjedi mali, tu je tvoj dom");
    must_keep("Ti ne vjeruj nikome na riječ");
    must_keep("Pjevaj Bosno, dušmani nek znaju da se lako ne damo");
    must_keep("Kapetane Laziću, narod tebe voli");
    must_keep("Bože dragi, što je lijepo kad se zaljubiš");
    must_keep("Kad svakog prolaznika ono nešto pita");

    if (failures) {
        std::printf("\nFAIL: %d problem(s) above\n", failures);
        return 1;
    }
    std::printf("\nok\n");
    return 0;
}