# Findings — Linetime / TaratorMusic, end of session

Written by the assistant for whoever picks this up. Everything below was measured on
this machine unless marked as assumed. Where I got something wrong, that is recorded
too, because it is the fastest way to avoid repeating it.

**Session 10 additions are marked `S10`.** They supersede section 13 where they
conflict with it: the two cheapest suggestions in section 13 were both measured and
one of them is now known to be actively harmful. See section 14.

## 1. Read this first: three claims I made that turned out wrong

These cost real time. Do not trust a single measurement without checking the whole set.

1. **"The DB language is wrong for `tomb`."** False. I read 4 chorus lines that happened
   to be English and generalised. Scored across all 69 GT lines, forced `bs` and `auto`
   both give 24/69 — identical. The DB language is fine.
2. **"The `-ot` flag crashes whisper."** False. `-ot` is `--offset-t`, which eats the next
   argument as a number, so `-ot -np` throws on `stoi`. My flag list was invented. Real
   word-timing flags are `-ojf` and `-otxt`, and Linetime already used `-ojf`.
3. **"cuDNN is missing, so GPU cannot work."** Half false. `TaratorMusic/bin/lib_gpu` really
   did lack it, but `sounddetect/vendor/cuda-12.6/lib/` had all 16 files the whole time. I
   checked one directory and generalised. I should have read `AGENTS.md`, which documents
   the GPU setup explicitly, including a prebuilt CUDA whisper-cli.

Also: I twice scored a run as catastrophic when the output file simply did not exist.
A scoring script must report missing inputs rather than counting them as zero.

## 2. Hardware and toolchain, verified

```
GPU:      NVIDIA GeForce GTX 1070, compute capability 6.1 (sm_61), 8192 MiB
ORT:      1.19.2
cuDNN:    9.x, 16 files in sounddetect/vendor/cuda-12.6/lib/
CUDA CLI: sounddetect/vendor/whisper.cpp/build-cuda/bin/whisper-cli  (prebuilt, works)
GPU libs: TaratorMusic/bin/lib_gpu/  (24 files, ~3.4 GB after cuDNN was copied in)
```

Note the arch: the CI GPU builds target `75;80;86;89;90`, which does **not** include 61.
That did not stop anything here — ORT and ggml-cuda carry their own arch handling — but do
not assume a CI-built binary will run on this card without checking.

`nvidia-smi` works, so GPU is genuinely available on this box.

## 3. GPU is now proven end to end

Both stages verified on the 1070 using **only the app's own folders**:

```
[transcriber] GPU confirmed: ... CUDA : ARCHS = 610 ...
[ctc] Execution provider: CUDA
CTC refinement: 16/17 lines re-timed
[00:33.38] Bože dragi što je lijepo kad se zaljubiš
```

The refusal path, verified at app level:

```
CPU CLI + --provider cuda  ->  "ran without CUDA... Refusing"
                            ->  exit 1
                            ->  no LRC file written
```

That is the behaviour the user asked for: a crash rather than a silent CPU swap.

**Two signals that look right and are not.** `systeminfo` in whisper's JSON contains the
word `CUDA` even on a CPU-only build, because it prints the template's CUDA line listing
what it *would* support. Checking for that string gives a false pass. Only
`ggml_cuda_init: found N CUDA devices` on the child's stderr is real. Proven by running the
same binary with and without the CUDA libraries reachable: 1 line versus 0.

**`-np` must not be passed to whisper.** It suppresses stderr, which carries the line the
GPU check depends on, so passing it made every GPU run look unverified. Per-segment
narration is filtered in `transcriber.cpp` instead.

## 4. The accuracy problem, measured

Method B produces **213 lines where ground truth has 270** — 21% short overall.

```
song    out   GT   ratio   within1s
tomb     36   69   0.52x   24/69
4arx     19   36   0.53x   12/36
5e1c     24   30   0.80x   14/30
ufx8     27   30   0.90x   18/30
ov4y     41   41   1.00x   32/41
cdeq     20   15   1.33x   11/15
0iac     46   33   1.39x   17/33
qfln     17   16   1.06x   15/16
```

Whisper emits one line per ~30s chunk, so a line can swallow 2-4 real lyrics. That is
Known Issue 10 and it is the top accuracy problem.

`text match` is near zero across the board because it demands exact character equality
and whisper drops diacritics. Do not use it as a headline number; compare side by side.

## 5. What was tried against it, and what actually worked

| approach | result | verdict |
|---|---|---|
| Language override (`Bosnian` -> `bs`) | tomb 6/16 -> 13/16 exact | **kept**, cheap, real |
| `[humming]` for wordless vocals | hum at 0:00/0:14 now `[humming]` | **kept** |
| Silero VAD | collapses on singing, see below | **rejected for singing** |
| `--separate-vocals` (centre channel) | segments cleaner, ratio unchanged | **kept, honestly labelled** |
| Multi-language merge `bs+en` | tomb 24 -> 25, never worse | **kept** |
| Blocklist hallucination check | passed synthetic tests, failed on real data | **discarded** |

**Silero VAD does not work on singing.** English speech gave 30 segments over a 152s track;
Bosnian songs collapsed to 2-7 segments covering a fraction of the audio. Whisper's own log:

```
n_chunks: 8498
chunk_len: 28 < n_window: 512
Merged 0 adjacent segments, now have 7 segments
```

Silero is trained on speech. Threshold and duration tuning changed nothing. Where it did
fire it was genuinely good, so this is a wrong-tool problem, not a broken tool.

**`--separate-vocals` is a mid/side subtraction, not a learned separator.** The centre
sits 10-15 dB above the sides on the test set, so it does drop accompaniment, and `tomb`
went from 7 raw whisper segments to 40. But CTC re-merges the lines, so the line ratio is
unchanged (0.49x with it, 0.52x without). Kept because it is free, not because it closes
the gap.

**The discarded blocklist check is worth remembering.** It scored every synthetic case
correctly and then classified real `Lavanda` as trustworthy, letting `Hvala što pratite
kanal` through. Whisper's confidence stays *high* on fabrications, so no confidence
threshold separates them. That is why hallucination work needs a model, not a word list.

## 6. Hallucinations — still the top unsolved problem

`qfln` opens with a fabricated `Hvala što pratite kanal` **even with `--separate-vocals`
on**. A credit line is a speech pattern, not a stereo position, so panning cannot touch
it. VAD would not see it either.

What has NOT been tried, cheapest first:

1. **`--no-speech-threshold` tuning on the whisper spawn.** Free, one flag. Try this before
   anything else.
2. **htdemucs in ONNX.** 316 MB single-file, RTF 0.20, ~8.8 dB median vocal SDR. A working
   C++ reference implementation exists (`sevagh/demucs.onnx`, ~34 KB across four files).
   The export deliberately puts STFT *outside* the graph, so C++ must do STFT, mel,
   normalisation, 7.8s segment inference, iSTFT and overlap-add itself. `htdemucs_ft` is
   1.26 GB and +0.4 dB, so do not start there.
   Our ORT is already linked and CUDA already works, so the provider side is solved.

The user was right that ffmpeg can do something here. It can: `pan="stereo|c0=c0|c1=-1*c1"`
removes a centred vocal. It cannot *extract* one — every "extract vocals" guide online
ends at that same dead end, and Audacity's equivalent only works when sources are already
on separate tracks. ffmpeg moves audio, it does not learn to recognise a singer.

## 7. Multi-language merge, shipped

`--language bs+en` transcribes each language and keeps the better result per segment,
scored on mean word probability with a small length reward and a 0.05 margin so near-ties
keep the primary. Per-segment is the whole point: comparing whole transcripts would let
the majority language veto the minority one, which is the case this exists for.

Measured, within 1s on GPU:

```
song     GT   single   merged
tomb     69   bs=24    bs+en=25  bs+auto=25
qfln     16   bs=15    bs+auto=15
5e1c     30   bs=14    bs+auto=14
8hfm     32   en=31    en+auto=31
```

`bs+auto` and `bs+en` score identically, so `auto` is the simpler pairing for the app.
Small gain, never a regression. The English it selects is still poor
(`Reapts higher` for `And re-inspire higher consciousness`) — that is transcription
quality, not a merge bug.

## 8. Ground truth is a reference, not an oracle

`synced_lyrics` is wrong in documented ways: per-song offsets, lines past audio EOF,
missing verses. Known Issues 1 and 5. Reading a GT disagreement as proof the tool is wrong
produced false alarms on `tgol`, `v2z4` and `mrz8`, where the DB was the broken party.

**Ask the user to hand-read the lyrics when a measurement is ambiguous.** Phrase the ask
concretely: song id, the seconds in question, the specific question. Do not silently
exclude a song to make a number look better.

## 9. Known-bad songs and what they are

```
qfln  Lavanda, Halid Beslic, Bosnian          hum intro 0:00-0:33, Turkish credit at tail
5e1c  Bosanska Artiljerija, Bosnian
cdeq  Zastava Bosanska, Bosnian
4arx  Balada o ratniku, Roki Vulovic, Serbian  5 Muzika marker lines
ov4y  Otkrit cu ti tajnu, Dino Merlin
tomb  Sredinom, Dino Merlin, mostly Bosnian    English chorus, GT all English words
ufx8  Kapetane Lazicu, Roki Vulovic, Serbian
0iac  Pavlovica cuprija, no language in DB
```

`tomb` is the mixed-language case. The user confirmed the chorus words (`Reach higher`,
`Higher consciousness`) are the real lyrics, and that it is "like a Korean song with a
few English words" — mostly Bosnian body, English chorus.

## 10. Repo state

Both worktrees clean except where noted. Neither has been pushed by me; I have no
credentials.

```
sounddetect (Linetime)   6 unpushed: 2ccb928 344b92e 9e038a4 dd3ff09 afd1792 08208eb
TaratorMusic             9 unpushed: f90de08 15d5d23 d4b3a9c 962f04a f911c3a b7d47e2 ...
```

**TaratorMusic has 2 uncommitted files** — the VAD plumbing that was proven inert:

```
backend/internal/appfiles/linetime.go   adds the "vad" component
backend/linetime_fetch/main.go          --vad-only, downloadVadModel, vadHF
```

The download works (verified, 885,098 bytes) and whisper accepts it. Decide whether to
keep it labelled speech-only or drop it before it reaches a Settings row implying it helps.

**The local `v1.3.0` tag still points at `86b9aed`,** which predates every fix here.
`git push --tags` would republish the broken commit. Delete or retarget before tagging 1.4.0.

**`AGENTS.md` in sounddetect is gitignored and untracked** (`.gitignore:28`), so it is
local-only and will not reach another clone. This file is tracked.

## 11. CI and release state

All five build jobs were green at last check. Triggers are now tags (`v*`), pull requests
and manual dispatch only, so a normal push starts nothing.

Release assets published per platform:
`linetime-{linux,windows}-x64-cpu.tar.gz`, `linetime-linux-x64-gpu.tar.gz`,
`linetime-macos-universal.tar.gz`, `linetime-windows-x64-gpu.zip`,
`linetime-whisper-cli-cuda-x64.tar.gz`, `linetime-whisper-cli-cuda-x64-windows.zip`.

The two CUDA CLI assets exist because **upstream whisper.cpp publishes no Linux CUDA
whisper-cli at all** — only Windows has a cublas build. Both CI jobs assert a real CUDA
link before packaging, so a build that lost CUDA support cannot ship as one.

**The app's CUDA CLI download has never run end to end**, because the asset does not exist
until a tag is pushed. Everything around it is verified: the resolution reports a clear
error when the release predates the asset, and GPU runs work from hand-installed folders.

## 12. Ground rules from the user

- Talk plainly, junior-dev level, short. Not caveman for its own sake.
- Report bad news early. They would rather have it now.
- Never claim a measurement from a partial read. Check the whole set.
- Silent CPU fallback is unacceptable. Crash instead.
- The todo list in the assistant tool is the progress tracker. Keep it current.
- The user reads `AGENTS.md` rarely and does not always know the codebase; surface
  relevant facts rather than assuming they know.

## 13. Suggested order next session

Items 1 and 2 were done in session 10 and item 1 turned out to be a dead end. See
section 14 before acting on this list.

1. ~~`--no-speech-threshold` tuning on the whisper spawn, measured on qfln.~~ **S10: done,
   no effect. Do not spend more time on it.**
2. ~~Re-run the 8-song set with whatever that yields, comparing against 213/270.~~ **S10:
   done, 229/270 with the marker filter in place.**
3. Decide the fate of the uncommitted VAD plumbing. Still open.
4. Delete or retarget the `v1.3.0` tag, then push, then tag 1.4.0. Still open.
5. Only then consider htdemucs, and measure plain 316 MB before the 1.26 GB bag.

## 14. Session 10: what was measured, including one plan that was wrong

### 14.1 `--no-speech-threshold` does nothing. The flag is real; the hope was not.

Now exposed as `--no-speech-threshold` / `-nth`, forwarded to whisper and verified to
reach the child (`-nth 0.420000` observed in the spawned argv via a recording wrapper).

Measured, method b on GPU, transcripts compared byte for byte:

```
song    qfln (hallucinating)   5e1c (hallucinating)
nth     0.05   0.30 0.60 0.80 0.95    0.05  0.95
result  identical in every case
```

The reason is visible in the source. `no_speech_thold` is read in exactly two places
(`vendor/whisper.cpp/src/whisper.cpp:7742` and `:7772`), and both require either a failed
temperature or a genuinely silent window. `Hvala što pratite kanal` is neither: whisper
decodes it confidently *and* believes it heard speech, so no threshold touches it. The same
argument covers `-sns`, which is already passed — the hallucination is ordinary words, not
a non-speech token.

This confirms the section-5 lesson a second time: **confidence stays high on fabrications,
so no threshold separates them.** Two thresholds have now been tried and both failed. The
next attempt needs a model.

Keep the flag anyway: it is a genuine knob for speech-heavy tracks, where it does bite. It
is exposed rather than hardcoded so that is not our decision to make silently.

### 14.2 Method b was writing stage directions as lyrics. Fixed.

`is_whisper_marker` lived as a `static` in `reconcile.cpp`, so method c used it and method b
did not — Known Issue 8. It now lives in `src/whisper_markers.h` and both paths call it.

One extra detail: `utils::normalize` transliterates Cyrillic, so `Музика` reached the
comparison as `muzika`, which the old Latin-only list never matched. The list now carries the
transliterated Serbian spellings, plus subtitling narration (`Субтитрујуће`).

Measured on the 8-song set, cached transcripts so only the filter differs:

```
4arx   2 marker lines ("Музика", "Субтитрујуће") -> [humming]
qfln 5e1c cdeq ufx8   byte-identical, no regression
```

### 14.3 The planned BoH / self-repetition layer would DELETE CORRECT OUTPUT

The plan in `AGENTS.md` proposes BoH removal, using "self-repeating transcript" as the
trigger for the trust check. **Repetition is not a hallucination signal on this test set.**
Counting lines repeated 3+ times in the *ground truth itself*:

```
tomb   51/69 lines (74%) are a repeated chorus, one line x11
cdeq   39%     ov4y 46%     5e1c 53%
ufx8   40%     qfln 50%     0iac 24%     4arx 17%
```

These songs are verses and choruses. A repetition trigger would fire on every one of the
eight and strip real lyrics. **Do not implement BoH as planned.** If a trust check is built,
it needs a signal that is absent from genuine choruses — word probability does not qualify,
and neither does repetition. This is the clearest result of the session and it cost one
query to establish.

### 14.4 Per-song offsets are NOT the dominant error, contrary to Known Issue 1

Known Issue 1 claims DB GT carries per-song constant offsets. Removing the per-song median
residual barely moves the total and makes two songs worse:

```
                     acc@0.5s  acc@1s  acc@2s
raw                    0.548    0.678   0.704
per-song offset removed 0.609    0.670   0.722
```

Only `4arx` has a real offset (+10.56 s median, 0.29 -> 0.57 once removed). The other seven sit
within ±0.9 s. So the per-song offset is a property of *one or two* broken DB entries, not a
systematic convention — which is consistent with section 8 and section 9.

### 14.5 Current method-B standing (8 songs, 270 GT lines)

```
song     GT   out  ratio  a@0.5s  a@1s  a@2s
qfln     16    17   1.06   1.000  1.000  1.000
5e1c     30    24   0.80   0.462  0.538  0.538
cdeq     15    20   1.33   0.500  0.500  0.500
4arx     36    21   0.58   0.286  0.286  0.286
ufx8     30    24   0.80   0.846  0.846  0.923
tomb     69    36   0.52   0.538  0.538  0.615
ov4y     41    41   1.00   0.433  0.867  0.867
0iac     33    46   1.39   0.250  0.333  0.417
TOTAL   270   229   0.85   0.548  0.678  0.704
```

229/270 lines against section 4's 213/270. Note these are *not* all comparable to section 4:
`--separate-vocals` and the marker filter both landed in between, so the comparison is
directional, not like-for-like.

### 14.6 A scoring mistake worth not repeating (the section-1 trap, again)

Method C on `8hfm` first scored **0.594 acc@1s, mean dt 2293 ms**. That was wrong. The tool
emits one extra line with empty text at a whisper gap, which shifted every later line by one
row; a naive index-aligned comparison then scored shifted rows against the wrong GT lines.
Re-scored with a DP alignment that tolerates spurious lines:

```
method A  8hfm  32/32  acc@1s 1.000  mean dt  19 ms
method C  8hfm  32/32  acc@1s 1.000  mean dt  19 ms
```

Both methods are fine. The harness was wrong, not the tool. **Any harness that compares by
line index needs a DTW-style alignment**, because the output line count never matches the GT
count exactly, in either direction. The `eval_b.py` harness in `/tmp/opencode` now does this.

### 14.7 Still open, unchanged from section 13

- VAD plumbing in TaratorMusic, 2 uncommitted files, proven inert on singing.
- `v1.3.0` tag still points at `86b9aed`, which predates every fix. Delete or retarget
  before tagging 1.4.0.
- 7 unpushed commits in sounddetect (was 6).
- htdemucs untested. It remains the only untried idea that addresses 14.1's actual cause.
---

## 15. Session 10b: htdemucs is out, and the narration filter that works instead

### 15.1 htdemucs is not viable on this GPU — do not retry it

`smank/htdemucs-onnx` loads on the CUDA EP correctly (GPU at 100%, 2.9 GB of 8 GB VRAM,
~1 CPU thread busy), and then a **single 7.8-second segment does not finish in 30 minutes**
on the GTX 1070. It was still running when killed; no output file was produced.

The input length is not a tuning knob. The graph contains a `Reshape` to
`{1,4,-1,343980}`, so any length that is not a multiple of 343980 frames (7.8 s at
44.1 kHz) fails:

| input | frames | result |
|---|---|---|
| 30 s | 1323000 | `Reshape` error on `/Reshape_20` |
| 8.0 s | 352800 | `Reshape` error on `/Reshape_20` |
| whole track, qfln 208 s | 9174529 | requests **41 GB** of activation memory |
| exactly 7.8 s | 343980 | runs, but >30 min, incomplete |

A 3.5-minute song is 27 segments, so even at a hypothetical 1 s/segment this would be fine,
but 30 min/segment is days per song. The user's call was that this cannot ship in an app
like this, and that is the right call. **Separation via htdemucs is closed.** The 304 MB
model is not the problem; the graph's compute on an sm_61 card is.

### 15.2 Five candidate hallucination signals, four rejected on measurement

The goal was a signal that is LOW on a fabrication and HIGH on a genuine repeated
chorus. Repetition itself cannot be the signal (14.3), so four alternatives were measured
against the cached transcripts and the 8-song ground truth. Medians over the same runs:

| signal | real (sung) | fabricated | verdict |
|---|---|---|---|
| whisper mean word probability | 0.936 | 0.846 | rejected — ranges overlap, and the credit lines themselves scored 0.761 / 0.857 |
| CTC forced-alignment confidence | 0.605 | 0.585 | rejected — no separation; caught only 1 of 3 |
| audio RMS over the segment | −18.96 dBFS | −19.06 dBFS | rejected — instrumental outros are as loud as singing |
| global repetition count | — | — | rejected in 14.3: 17–74% of each song's GT is real chorus |
| consecutive near-duplicate runs | — | — | rejected — see below |

The last one was the most promising idea in this session, and the ground truth killed it.
Counting the longest run of back-to-back near-duplicate lines (Jaccard ≥ 0.5) in every
song with synced GT:

```
tomb 16   wnkf 9   vugk 9   v2z4 7   xm8u 4   tnz0 4   ybyi 4   opmx eight runs of 3
```

26 songs legitimately repeat a line back-to-back, the worst being **tomb at 16 consecutive
lines**. So "whisper printed the same line three times in a row" is not a fabrication
signal; `tomb` would lose 16 real lyric lines. Measured on the actual method-B output the
signal also barely fires: max consecutive-duplicate run was 2 on all four songs tested.

Note the trap here. 5e1c's output contains "Artiljerija, Bosanac sam bekrija" three times
and it *looks* like a loop. It is not — that is the real chorus, and the GT has it four
times. Two intermediate versions of the labelling harness in this session called that
chorus a fabrication before a whole-song vocabulary match corrected it. Judging a line
"real" by matching one neighbouring GT line is wrong for exactly the songs that matter.

### 15.3 What works: a closed-class narration phrase list

The one signal that separates cleanly is categorisation rather than measurement. Whisper's
broadcast credit line ("Hvala što pratite kanal.", "Hvala na sviđanju!") is *narration
about the video*, and nobody sings that. So it can be matched lexically, and the
false-positive rate — the number that actually decides whether this is safe — was measured
against the database rather than assumed:

```
8-song test set        0 hits / 270 GT lines   (0.000%)
all synced GT in DB    1 hit  / 2431 GT lines  (0.041%)
```

The single hit is `g1um` at 03:29.90, whose ground truth is *itself* "Hvala što pratite
kanal." — the same hallucination already baked into the reference. That is a small but
real piece of evidence that `synced_lyrics` carries whisper artefacts, not just sync
offsets, and it is worth knowing before treating any GT line as settled.

Shipped in `91492f0`. Result on all 8 songs, replaying cached transcripts:

| song | before | after |
|---|---|---|
| qfln | `[00:00.00] Hvala što pratite kanal.` | `[00:00.00] [humming]` |
| 5e1c | `[03:18.68] Hvala što pratite kanal.` | `[03:17.70] [humming]` |
| 0iac | `[04:01.16] Hvala na sviđanju!` | `[04:01.16] [humming]` |
| cdeq, 4arx, ufx8, tomb, ov4y | — | byte-identical |

No timing movement: qfln and 0iac timestamps are identical, 5e1c moves at most 2.62 s.
Regression on 8hfm, old binary vs new: method A and method C both byte-identical, both
acc@1s 1.000, mean dt 19 ms.

Two entries in the list were wrong in a way that review did not catch and only re-reading
the output did:

- `s-v-i-đ-a-n-j-u` normalizes to **`svidanju`** (eight letters, one j). Written with two
  j's, the filter silently missed 0iac.
- `normalize()` keeps apostrophes and maps U+2019 to `'`, so the entry is **`don't forget
  to`**, not `dont forget to`.

`tools/narration_probe.cpp` now asserts the whole list in milliseconds with no GPU, model
or audio. Build: `g++ -std=c++17 tools/narration_probe.cpp -Isrc -pthread`. Lesson worth
keeping: for a hand-written match list, the test is "does every entry survive its own
normalizer", and that test is cheap enough to always run.

### 15.4 What is still broken, stated plainly

**Correction to this section as it stood earlier in the session: 5e1c is NOT a repetition
loop.** It appeared in three separate lists as one. What it actually does is place two
chorus lines in the outro (184.5 s and 201.3 s) where the vocals end at 167.9 s and the
audio runs to 216.1 s — so it over-*reaches* at the tail, while under-producing repeats
elsewhere (24 output lines against 29 GT lines). Its chorus "Artiljerija, Bosanacam
Bekrija" looks like a loop because the GT has that line four times; judging a whisper line
real by matching one neighbouring GT line is exactly the mistake 15.2 warns about.

So the outstanding repetition-loop class is **cdeq only**, and one song is a much weaker
claim than three. The remaining failures transcribe chorus text that genuinely is being
sung, so neither the audio nor whisper's confidence distinguishes them from a chorus sung
four times. Everything cheap has now been measured and rejected (15.2).

## 16. The database ground truth is verified, not just "a reference"

§8 and Known Issue 1 both treat `synced_lyrics` as untrustworthy. For these songs that is
no longer true — it was cross-checked against **LRCLIB**, an independent synced-lyrics
database, at `https://lrclib.net/api/search` and `/api/get` (send a `User-Agent`; it
returns 503 often enough to need backoff).

| song | LRCLIB | vs DB |
|---|---|---|
| qfln, 5e1c, ufx8, 0iac | plain lyrics only | text identical on all lines |
| tomb, ov4y, 8hfm | synced | text identical, and **timing identical** |
| cdeq, 4arx | nothing found | DB only |

- **Text: 100% line-for-line agreement on all 7 songs LRCLIB had.** The DB's lyrics are
  correct, not merely plausible.
- **Timing: tomb and ov4y match LRCLIB exactly (median difference 0.00 s, 69/69 and 41/41
  rows identical).** 8hfm agrees within 1 s on all 32 rows. This retires Known Issue 1's
  per-song offset story *for those songs* — ov4y's median residual is +0.48 s, not an
  offset. The offsets in Known Issue 1 were measured against tgol/v2z4/mrz8, which were
  separately confirmed DB-broken; they do not generalise.
- `0iac`'s last GT line is `[04:00.66] Via Rade Ilic` — a *credit line*, same class as
  "Hvala što pratite kanal." Two independent confirmations now that `synced_lyrics`
  carries whisper artefacts, since GT cannot be used to score the narration filter's
  output as if every GT line were a lyric.
- cdeq and 4arx have no LRCLIB entry. Their DB rows are unverified. Do not quote a cdeq
  timing number as measured against truth.

## 17. Method B's real defect was not hallucination — it was split lines

Method B's headline accuracy was being blamed on hallucination. Labelling every output
line against GT showed most of the damage was something else entirely: **whisper splits a
lyric line and emits the tail as its own segment.** 46 of 242 lines on the 8-song set.

```
[00:55.28] Godinama jedna jabuka zrela, podijeli na dugo na dva ista
[01:01.80] dijela                          <- same lyric, two LRC entries
[01:26.56] lete.
[00:29.10] ale druge
```

Ground truth contains no one- or two-word line anywhere, so these were never
hallucinations. They are pure output damage — an LRC no karaoke player can use. Fixed in
`815f0a8` by rejoining a segment of ≤2 words into the line above it. All 39 merges were
checked against GT: **none lost coverage of the line it belonged to.**

| 8 songs, method B | before | after |
|---|---|---|
| text accuracy | 0.353 | **0.383** |
| acc@0.5s | 0.410 | **0.497** |
| acc@1s | 0.500 | **0.601** |
| acc@2s | 0.563 | **0.645** |

No song regressed. Methods A and C on 8hfm are unchanged at 1.000.

### 17.1 Three merge signals, two of them wrong

| signal | merges | why it fails |
|---|---|---|
| previous line does not end in punctuation | 196/234 | almost every boundary looks unfinished. Collapses 242 lines to 46 |
| current segment starts lowercase | 39 | **correct and all 39 were sound** — but the check is ASCII, and Serbian returns in Cyrillic whose leading byte is neither case. 4arx collapsed 21 lines → 9 |
| current segment is ≤ 2 words | 39 | **used.** Script-agnostic, needs no case table |

The lowercase signal is genuinely good and is the one to reach for if this is revisited
with a real Unicode case table — do not re-derive that it does not work, it does.

The time gap between segments is useless as well: genuine line breaks routinely have a
0.00 s gap.

Two bugs found while validating, both worth not repeating:

- The repeat guard (`"don't glue a repeated refrain"`) first used `strstr`. The fragment
  `"ela"` is the tail of `"zrela"`, so it matched inside that word and refused a merge
  that was obviously right. Compare **whole words**, not substrings.
- `is_line_tail` counts words with `isspace`-style splitting. `std::isspace` on a signed
  char with a high bit set is UB; splitting on an explicit `' '`/`'\t'` test avoids it and
  matches what the Python A/B tool does.

### 17.2 A trustworthy scorer, and why the old numbers must not be compared to these

`tools/score_b.py` (new, replaces the lost `eval_b.py`). Monotone DP, one predicted line
may absorb a run of consecutive GT lines, timing scored only on the GT line that *starts*
a segment. Transliterates Cyrillic and strips diacritics first — comparing raw strings
scores correct Serbian as wrong.

**The Session 9 figures (acc 0.63–0.88, 300 predicted lines) are not reproducible and do
not use this definition.** Their per-song predicted-line counts do not match the same
build's output at all (`ov4y` 77 predicted lines when the tool wrote 41), so they were
computed over a different unit. Session 10b's `score_b.py` was also broken — it compared a
single line's containment against a *cumulative* DP score, so single-line blocks were
almost never chosen and every alignment came out shifted by two lines.

Two numbers to keep in mind when reading any accuracy figure:

- **Oracle ceiling for the pre-fix output was 0.502** — the best containment any GT line
  could reach from any predicted line, ignoring order entirely. Read a method-B `acc`
  against that, not against 1.0.
- **Do not tune a merge rule on `acc`.** It rewards over-merging: 46 enormous blob lines
  score 0.565, higher than any honest output. Report fragment count (lines of ≤2 words)
  alongside it, or the rule will be optimised straight into rubble.
