#!/usr/bin/env python3
"""Check Part B contrast versions.

  python3 check_contrast.py --ref <slice.jsonl> --out <contrast.jsonl>

Per row: ids in slice order and split/move copied; `formal` 1–3 sentences, no contractions,
no exclamation marks, no blocklisted fragments; `near_miss` 1–5 sentences, <= 60 words, no blocklisted
fragments; every clock time, number, weekday, month and mid-sentence name of the source appears in both
versions (same rules as the fact section of validate_titan.py).
"""
import sys, re, json, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "titan_v1", "raw")]   # validate_titan.py lives in titan_v1/raw
from validate_titan import (time_ok, num_words, has_any, cardinal, WEEKDAYS, MONTHS, BLOCKLIST,
                            CONTRACTION_RE, sentences)


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fact_issues(src, text, label):
    low = text.lower().replace("-", " ")
    issues = []
    rest = src
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\b", src):
        if not time_ok(m.group(1), m.group(2), low):
            issues.append(f"{label}: time '{m.group(0)}' missing")
        rest = rest.replace(m.group(0), " ")
    for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?", rest):
        tok = m.group(0).replace("$", "").replace(",", "")
        if "." in tok:
            whole, frac = tok.split(".")
            ok = tok in low or (tok == "0.25" and "quarter" in low) or (tok == "0.5" and "half" in low) or \
                 (has_any(low, num_words(int(whole))) and has_any(low, num_words(int(frac))))
        else:
            ok = tok in low or has_any(low, num_words(int(tok)))
        if not ok:
            issues.append(f"{label}: number '{m.group(0)}' missing")
    for d in WEEKDAYS + MONTHS:
        if d == "may":
            present = re.search(r"(\d\w*\s+May\b|\bMay\s+\d|\b(in|by|until|of|before|after|this|next|last)\s+May\b)", src)
        else:
            present = re.search(rf"\b{d}\b", src, re.I)
        if present and d not in low:
            issues.append(f"{label}: '{d}' missing")
    for m in re.finditer(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b", src):
        name = m.group(1)
        if name.lower() in WEEKDAYS + MONTHS:
            continue
        if name.lower() not in low:
            issues.append(f"{label}: name '{name}' missing")
    for b in BLOCKLIST:
        if b in low:
            issues.append(f"{label}: blocklisted fragment '{b}'")
    return issues


def main():
    a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    ref, out = read(a["--ref"]), read(a["--out"])
    problems = []
    if [r["id"] for r in ref] != [o.get("id") for o in out]:
        problems.append(f"id sequence differs ({len(ref)} ref, {len(out)} out)")
    refmap = {r["id"]: r for r in ref}
    for o in out:
        r = refmap.get(o.get("id"))
        if not r:
            problems.append(f"{o.get('id')}: not in reference"); continue
        tag = o["id"]
        for k in ("split", "move"):
            if o.get(k) != r[k]:
                problems.append(f"{tag}: {k} {o.get(k)!r} != {r[k]!r}")
        f, n = o.get("formal", ""), o.get("near_miss", "")
        if not f or not n:
            problems.append(f"{tag}: empty formal or near_miss"); continue
        nf = len(sentences(f))
        if not 1 <= nf <= 3:
            problems.append(f"{tag}: formal has {nf} sentences (1–3)")
        if CONTRACTION_RE.search(f):
            problems.append(f"{tag}: formal has a contraction")
        if "!" in f:
            problems.append(f"{tag}: formal has an exclamation mark")
        nn, wn = len(sentences(n)), len(n.split())
        if not 1 <= nn <= 5:
            problems.append(f"{tag}: near_miss has {nn} sentences (1–5)")
        if wn > 60:
            problems.append(f"{tag}: near_miss has {wn} words (>60)")
        if f.strip().lower() == r["source"].strip().lower() or n.strip().lower() == r["target"].strip().lower():
            problems.append(f"{tag}: a version copies the source or the Titan target")
        problems += fact_issues(r["source"], f, f"{tag} formal")
        problems += fact_issues(r["source"], n, f"{tag} near_miss")
    for p in problems[:60]:
        print("PROBLEM", p)
    print(f"rows: {len(out)}  problems: {len(problems)}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
