#!/usr/bin/env python3
"""Offline A/B for Method B line-merging, using cached whisper transcripts.

Whisper splits a lyric line and then emits the remainder as its own segment:
"Godinama jedna jabuka zrela, podijeljena tugom na dva" followed by "dijela."
Ground truth has no such short lines, so those are not hallucinations and not
whisper errors -- they are one lyric split across two LRC entries, which is
Known Issue 10. This applies candidate merge rules to the cached transcript and
scores each result, so no GPU run is needed to compare rules.

A merge is only safe when the previous line does not look finished. The signal is
the trailing character of the previous segment, because that is what tells you
whether whisper stopped at a phrase boundary or ran out of window mid-line.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_b import NONLYRIC, norm, read_gt, score  # noqa: E402

import sqlite3  # noqa: E402


def ends_sentence(t):
    return bool(t) and t.rstrip()[-1] in '.!?…'


def ends_clause(t):
    return bool(t) and t.rstrip()[-1] in '.!?…;:'


def merge(segments, rule):
    """segments: list of (start_s, text). Returns merged list."""
    out = []
    for t, text in segments:
        text = text.strip()
        if not text:
            continue
        if out and rule(out[-1][1], text):
            pt, ptext = out[-1]
            out[-1] = (pt, ptext + ' ' + text)
        else:
            out.append((t, text))
    return out


def starts_lower(t):
    t = t.lstrip()
    return bool(t) and t[0].islower()


def words(t):
    return len(norm(t).split())


def repeats(prev, cur):
    """Whisper looping a short refrain ('O Bože' four times) is a different thing
    from whisper splitting a line. Merging that would glue four real lines together
    and lose their timings, so a continuation that merely repeats text the previous
    line already has is left alone.

    Compared as whole words, not as a substring: the fragment 'ela' is the tail of
    'zrela', and a plain substring search finds it inside that word."""
    c, p = norm(cur).split(), norm(prev).split()
    if not c:
        return False
    return any(p[i:i + len(c)] == c for i in range(len(p) - len(c) + 1))


def low_or_short(n):
    """A continuation either starts lowercase -- whisper did not begin a new
    sentence, so this is the tail of the line it was already writing -- or is short
    enough that it cannot be a whole lyric line."""
    if n <= 0:
        return lambda prev, cur: False
    return lambda prev, cur: (starts_lower(cur) or words(cur) <= n) and not repeats(prev, cur)


RULES = {
    'none':      lambda prev, cur: False,
    'sent':      lambda prev, cur: not ends_sentence(prev),
    'lower':     lambda prev, cur: starts_lower(cur),
    'short2':    lambda prev, cur: words(cur) <= 2 and not repeats(prev, cur),
    'short3':    lambda prev, cur: words(cur) <= 3 and not repeats(prev, cur),
    'final':     low_or_short(2),
    'lower|4w':  low_or_short(4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='/mnt/guncelprojeler/TaratorMusic/taratordb/musics.db')
    ap.add_argument('--transcripts', default='tcache/j')
    ap.add_argument('songs', nargs='+')
    a = ap.parse_args()
    db = sqlite3.connect(a.db)
    cache = {}
    for sid in a.songs:
        d = json.load(open(f'{a.transcripts}/tarator-{sid}.json'))
        segs = [(s['offsets']['from'] / 1000.0, s['text']) for s in d['transcription']]
        cache[sid] = segs

    print(f"{'rule':<11} {'lines':>6} {'ok':>4} {'acc':>6} {'timed':>6} {'@0.5':>6} {'@1':>6} {'@2':>6}")
    for name, rule in RULES.items():
        T = {'gt': 0, 'ok': 0, 'n': 0, 'c05': 0, 'c1': 0, 'c2': 0}
        lines = 0
        for sid in a.songs:
            gt = read_gt(db, 'tarator-' + sid)
            pred = []
            for t, text in merge(cache[sid], rule):
                w = norm(text)
                pred.append((t, '' if w in NONLYRIC else w))
            lines += len(pred)
            r = score(sid, gt, pred)
            if r is None:
                continue
            T['gt'] += r['gt']; T['ok'] += r['ok']; T['n'] += len(r['resid'])
            for k, th in (('c05', 0.5), ('c1', 1.0), ('c2', 2.0)):
                T[k] += sum(1 for x in r['resid'] if abs(x) <= th)
        n = max(T['n'], 1)
        print(f"{name:<11} {lines:>6} {T['ok']:>4} {T['ok']/max(T['gt'],1):>6.3f} {T['n']:>6} "
              f"{T['c05']/n:>6.3f} {T['c1']/n:>6.3f} {T['c2']/n:>6.3f}")


if __name__ == '__main__':
    main()