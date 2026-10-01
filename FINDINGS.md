# Findings — Linetime / TaratorMusic, end of session

Written by the assistant for whoever picks this up. Everything below was measured on
this machine unless marked as assumed. Where I got something wrong, that is recorded
too, because it is the fastest way to avoid repeating it.

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

1. `--no-speech-threshold` tuning on the whisper spawn, measured on qfln. Cheapest shot at
   hallucinations and nothing else has touched them.
2. Re-run the 8-song set with whatever that yields, comparing against 213/270.
3. Decide the fate of the uncommitted VAD plumbing.
4. Delete or retarget the `v1.3.0` tag, then push, then tag 1.4.0.
5. Only then consider htdemucs, and measure plain 316 MB before the 1.26 GB bag.