#!/usr/bin/env python3
"""Checks for Titan Style Signals deliverables.

  python3 check_style_v1.py scores --ref <rows.jsonl> --scores <attributes.jsonl> [--targets-only]
      ids must match the reference rows in order; 7 keys; values in {0.0,…,1.0}; prints separation stats.
  python3 check_style_v1.py rescore --first <attributes.jsonl> --second <rescore.jsonl>
      mean absolute difference per attribute on the ids present in both (target_style).
"""
import sys, json

KEYS = ["formality", "grandiosity", "dramatic_tone", "philosophical_tone",
        "authoritative_tone", "elevated_vocabulary", "aggression"]


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def valid_vec(v, where, problems):
    if not isinstance(v, dict):
        problems.append(f"{where}: not an object"); return
    for k in KEYS:
        if k not in v:
            problems.append(f"{where}: missing {k}"); continue
        x = v[k]
        if not isinstance(x, (int, float)) or x < 0 or x > 1 or abs(round(x * 10) - x * 10) > 1e-6:
            problems.append(f"{where}: {k}={x!r} not a 0.1 step in 0..1")
    extra = set(v) - set(KEYS)
    if extra:
        problems.append(f"{where}: extra keys {sorted(extra)}")


def cmd_scores(a):
    ref, scores = read(a["--ref"]), read(a["--scores"])
    targets_only = "--targets-only" in a
    problems = []
    if [r["id"] for r in ref] != [s.get("id") for s in scores]:
        problems.append(f"id sequence differs from reference ({len(ref)} ref rows, {len(scores)} score rows)")
        ids_ref, ids_sc = [r["id"] for r in ref], [s.get("id") for s in scores]
        missing = [i for i in ids_ref if i not in set(ids_sc)]
        extra = [i for i in ids_sc if i not in set(ids_ref)]
        if missing: problems.append(f"  missing: {missing[:10]}{'…' if len(missing) > 10 else ''}")
        if extra: problems.append(f"  extra: {extra[:10]}{'…' if len(extra) > 10 else ''}")
    splits = {r["id"]: r.get("split") for r in ref}
    for s in scores:
        if s.get("split") != splits.get(s.get("id")):
            problems.append(f"{s.get('id')}: split {s.get('split')!r} != {splits.get(s.get('id'))!r}")
        if not targets_only:
            valid_vec(s.get("source_style"), f"{s.get('id')} source_style", problems)
        valid_vec(s.get("target_style"), f"{s.get('id')} target_style", problems)
    for p in problems[:40]:
        print("PROBLEM", p)
    print(f"rows: {len(scores)}  problems: {len(problems)}")
    if not targets_only and scores and not problems:
        print(f"{'attribute':<22}{'src mean':>9}{'tgt mean':>9}{'tgt>src':>9}")
        for k in KEYS:
            sm = sum(s["source_style"][k] for s in scores) / len(scores)
            tm = sum(s["target_style"][k] for s in scores) / len(scores)
            gt = sum(s["target_style"][k] > s["source_style"][k] for s in scores) / len(scores)
            print(f"{k:<22}{sm:>9.2f}{tm:>9.2f}{gt:>9.0%}")
    return 1 if problems else 0


def cmd_rescore(a):
    first = {s["id"]: s["target_style"] for s in read(a["--first"])}
    second = read(a["--second"])
    problems = []
    for s in second:
        valid_vec(s.get("target_style"), f"{s.get('id')} target_style", problems)
        if s.get("id") not in first:
            problems.append(f"{s.get('id')} not in first pass")
    for p in problems[:40]:
        print("PROBLEM", p)
    both = [s for s in second if s.get("id") in first]
    print(f"rescored rows: {len(both)}  problems: {len(problems)}")
    print(f"{'attribute':<22}{'MAD':>6}{'max':>6}{'within 0.1':>12}")
    worst = 0
    for k in KEYS:
        diffs = [abs(s["target_style"][k] - first[s["id"]][k]) for s in both]
        mad = sum(diffs) / len(diffs)
        worst = max(worst, mad)
        print(f"{k:<22}{mad:>6.3f}{max(diffs):>6.1f}{sum(d <= 0.1 + 1e-9 for d in diffs) / len(diffs):>12.0%}")
    print("PASS (all <= 0.15)" if worst <= 0.15 else "FAIL (an attribute exceeds 0.15)")
    return 1 if problems else 0


if __name__ == "__main__":
    args = sys.argv[2:]
    a = {args[i]: (args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True)
         for i in range(len(args)) if args[i].startswith("--")}
    sys.exit({"scores": cmd_scores, "rescore": cmd_rescore}[sys.argv[1]](a))
