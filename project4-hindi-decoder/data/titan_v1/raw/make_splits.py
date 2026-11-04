#!/usr/bin/env python3
"""Merge titan_pairs_*.jsonl, make stratified train/val/test splits, and render training views.

Outputs (in the same directory):
  titan_all.jsonl                      merged, validated rows
  titan_train.jsonl / titan_val.jsonl / titan_test.jsonl
  titan_sft_train.jsonl (+ val/test)   chat-format rows for the style LoRA
  titan_jepa_train.jsonl (+ val/test)  {"text": source, "code": target} rows for the LLM-JEPA view
Usage: python3 make_splits.py <dir> [--val 100] [--test 100] [--seed 13]
"""
import sys, os, glob, json, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_titan import load, check_row  # noqa: E402

INSTRUCTION = "Rewrite in the Titan's voice: "


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0] if args else "."
    opt = {k: int(v) for k, v in zip(sys.argv[1::1], sys.argv[2::1]) if k in ("--val", "--test", "--seed")}
    n_val, n_test, seed = opt.get("--val", 100), opt.get("--test", 100), opt.get("--seed", 13)

    rows, errors = load(sorted(glob.glob(os.path.join(d, "titan_pairs_*.jsonl"))))
    bad = [r["id"] for r in rows if check_row(r)]
    if errors or bad:
        sys.exit(f"fix validator issues first: {errors} {bad}")
    rows.sort(key=lambda r: r["id"])
    for r in rows:
        r.pop("_file", None)
    with open(os.path.join(d, "titan_all.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stratify by domain x move: take val/test proportionally from every stratum
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["domain"], r["move"])].append(r)
    val, test, train = [], [], []
    share_val, share_test = n_val / len(rows), n_test / len(rows)
    carry_v = carry_t = 0.0
    for key in sorted(strata):
        grp = strata[key][:]
        rng.shuffle(grp)
        carry_v += len(grp) * share_val
        carry_t += len(grp) * share_test
        kv, kt = int(carry_v), int(carry_t)
        carry_v -= kv
        carry_t -= kt
        val += grp[:kv]
        test += grp[kv:kv + kt]
        train += grp[kv + kt:]
    print(f"rows {len(rows)} -> train {len(train)}  val {len(val)}  test {len(test)}")

    def write(name, items):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for split, items in (("train", train), ("val", val), ("test", test)):
        items = sorted(items, key=lambda r: r["id"])
        write(f"titan_{split}.jsonl", items)
        write(f"titan_sft_{split}.jsonl", [
            {"id": r["id"], "messages": [
                {"role": "user", "content": INSTRUCTION + r["source"]},
                {"role": "assistant", "content": r["target"]}]} for r in items])
        # LLM-JEPA view: 'text' is the plain sentence (encoder input + [PRED] tokens), 'code' the styled target
        write(f"titan_jepa_{split}.jsonl", [
            {"id": r["id"], "text": r["source"], "code": r["target"]} for r in items])


if __name__ == "__main__":
    main()
