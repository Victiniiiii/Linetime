#!/usr/bin/env python3
"""Score method-B output against the database ground truth.

Two numbers per song, kept apart on purpose:

  acc    text quality. For every ground-truth line, the fraction of its words that
         appear in the predicted line the aligner assigned to it. `acc` counts a GT
         line only when containment is >= 0.80, so a line that is half-recited
         counts against the score.

  acc@1s timing quality. Scored only on the GT line that *starts* a predicted
         segment, because a 30 s whisper segment legitimately covers several lyric
         lines and the ones inside it have no single start time of their own.

The alignment is a monotone DP, not greedy per-line matching. Greedy matching is
wrong on songs with a repeated chorus: it pairs a prediction with whichever GT
line happens to contain it best, which is usually the wrong occurrence, and the
residuals that come out are fiction. This scorer lets one predicted line absorb a
run of consecutive GT lines and never goes backwards.

Transliteration and diacritics are stripped before any comparison. Whisper
returns Serbian in Cyrillic against Latin ground truth and drops the
Bosnian/Serbian diacritics (š ž ć đ), so comparing raw strings scores correct
lyrics as wrong.
"""

import argparse
import json
import re
import sqlite3
import sys
import unicodedata

CYR = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'ch', 'ш': 'sh', 'щ': 'sh', 'ъ': '',
    'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}
TAG = re.compile(r'^\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$')
NONLYRIC = {'humming', 'music', 'muzika', 'applause', 'laughing', 'smiling'}


def norm(text):
    text = ''.join(CYR.get(c.lower(), c) for c in text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text.lower()).split())


def read_lrc(path):
    out = []
    for line in open(path, errors='replace'):
        m = TAG.match(line.strip())
        if m:
            t = int(m.group(1)) * 60 + float(m.group(2))
            words = norm(m.group(3))
            # A placeholder is not a lyric. Keeping it lets the timing of a silent
            # passage be scored, but it must not be credited as transcribed text.
            if words in NONLYRIC:
                words = ''
            out.append((t, words))
    return out


def read_gt(db, song_id):
    row = db.execute(
        "SELECT synced_lyrics FROM lyrics WHERE song_id=?", (song_id,)
    ).fetchone()
    if not row or not row[0]:
        return []
    out = []
    for line in row[0].splitlines():
        m = TAG.match(line.strip())
        if m:
            t = int(m.group(1)) * 60 + float(m.group(2))
            words = norm(m.group(3))
            if words and words not in NONLYRIC:
                out.append((t, words))
    return out


def contain(gt_words, pred_words):
    if not gt_words:
        return 0.0
    pool = list(pred_words)
    hit = 0
    for w in gt_words:
        if w in pool:
            hit += 1
            pool.remove(w)
    return hit / len(gt_words)


def align(gt, pred):
    """Monotone DP. Returns (owner[i], block_start[j]) where owner[i] is the
    predicted line that carries GT line i, and block_start[j] is the index of the
    first GT line covered by predicted line j (or None if unused)."""
    m, n = len(gt), len(pred)
    gt_words = [w.split() for _, w in gt]
    pr_words = [w.split() for _, w in pred]

    # best[j][i] = best score covering GT 1..i with pred 1..j.
    # A block covering GT a+1..i under pred j scores the *mean* containment of the
    # block's GT lines, so a long block is not rewarded for being long.
    NEG = -1e9
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            # Predicted line j may cover nothing at all, which lets a placeholder
            # or an extra predicted line sit in the timeline without forcing a GT
            # line onto it.
            if best[j - 1][i] > best[j][i]:
                best[j][i] = best[j - 1][i]
                back[j][i] = (j - 1, i)
            # Block covering GT a+1..i scores the *mean* containment of the block's
            # GT lines, so a long block is not rewarded merely for being long.
            # A placeholder prediction has no words, so every block scores 0 and
            # the pass-through above wins instead.
            run = 0.0
            for a in range(i - 1, -1, -1):
                run += contain(gt_words[a], pr_words[j - 1])
                cand = best[j - 1][a] + (run / (i - a))
                if cand > best[j][i]:
                    best[j][i] = cand
                    back[j][i] = (j - 1, a)
    if best[n][m] <= NEG / 2:
        return None, None

    owner = [None] * m
    block_start = [None] * n
    i, j = m, n
    while i > 0 and j > 0:
        prev = back[j][i]
        if prev is None:
            break
        pj, pa = prev
        block_start[j - 1] = pa
        for k in range(pa, i):
            owner[k] = j - 1
        i, j = pa, pj
    return owner, block_start


def score(sid, gt, pred):
    if not gt or not pred:
        return None
    owner, block_start = align(gt, pred)
    if owner is None:
        return None
    gt_words = [w.split() for _, w in gt]
    pr_words = [w.split() for _, w in pred]
    ok = part = miss = 0
    for i in range(len(gt)):
        j = owner[i]
        c = contain(gt_words[i], pr_words[j]) if j is not None else 0.0
        if c >= 0.80:
            ok += 1
        elif c >= 0.50:
            part += 1
        else:
            miss += 1
    resid = []
    for j, a in enumerate(block_start):
        # a == len(gt) means this predicted line was passed through: it covers no
        # GT line and therefore has no start time to score against.
        if a is None or a >= len(gt) or not pr_words[j]:
            continue
        resid.append(pred[j][0] - gt[a][0])
    return {
        'gt': len(gt), 'pred': len(pred), 'ok': ok, 'part': part, 'miss': miss,
        'acc': ok / len(gt),
        'resid': resid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='/mnt/guncelprojeler/TaratorMusic/taratordb/musics.db')
    ap.add_argument('--out', required=True, help='directory of <id>.lrc files')
    ap.add_argument('songs', nargs='+')
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    tot = {'gt': 0, 'ok': 0, 'part': 0, 'miss': 0, 'n': 0, 'c05': 0, 'c1': 0, 'c2': 0}
    for sid in a.songs:
        r = score(sid, read_gt(db, 'tarator-' + sid), read_lrc(f'{a.out}/{sid}.lrc'))
        if r is None:
            print(f'{sid:<6} SKIP (no gt or no prediction)')
            continue
        rs = r['resid']
        f = (lambda t: sum(1 for x in rs if abs(x) <= t) / len(rs)) if rs else float('nan')
        print(f"{sid:<6} GT {r['gt']:>3}  pred {r['pred']:>3}  ok {r['ok']:>3} part {r['part']:>3} "
              f"miss {r['miss']:>3}  acc {r['acc']:.3f}  | timed {len(rs):>3}  "
              f"@0.5 {f(0.5):.3f}  @1 {f(1.0):.3f}  @2 {f(2.0):.3f}  "
              f"med {sorted(rs)[len(rs)//2]:+.2f}s" if rs else
              f"{sid:<6} GT {r['gt']:>3}  pred {r['pred']:>3}  acc {r['acc']:.3f}  | no timed lines")
        tot['gt'] += r['gt']; tot['ok'] += r['ok']; tot['part'] += r['part']
        tot['miss'] += r['miss']; tot['n'] += len(rs)
        for t, k in ((0.5, 'c05'), (1.0, 'c1'), (2.0, 'c2')):
            tot[k] += sum(1 for x in rs if abs(x) <= t)
    n = len(a.songs)
    print(f"\nTOTAL over {n}: GT {tot['gt']}  ok {tot['ok']}  part {tot['part']}  miss {tot['miss']}  "
          f"acc {tot['ok']/max(tot['gt'],1):.3f}")
    if tot['n']:
        print(f"timed {tot['n']}  @0.5 {tot['c05']/tot['n']:.3f}  @1 {tot['c1']/tot['n']:.3f}  "
              f"@2 {tot['c2']/tot['n']:.3f}")


if __name__ == '__main__':
    sys.exit(main())