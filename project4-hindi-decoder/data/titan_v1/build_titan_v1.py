"""Builds titan_v1 from the delivered Titan Register dataset (raw/) plus 100 handwritten pairs.

raw/titan_{train,val,test}.jsonl are the delivered splits (make_splits.py: stratified by domain x move, seed 13;
1,360 pairs, ids 1-240, 401-1200, 1361-1680). raw/handwritten_h301_h400.jsonl adds 100 pairs written by hand in the
same register (style guide v1), all passing validate_titan.py, with no input close to a validation, test or
demo sentence. They go to train only. Output rows use the project's schema (id, source_id, split, persona_id,
source, target, references) and keep the Titan tags (domain, intent, move, length, register).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def load(name):
    return [json.loads(l) for l in open(RAW / name, encoding="utf-8") if l.strip()]


def row(r, split, origin):
    rid = r["id"] if isinstance(r["id"], str) else f"titan_{r['id']:04d}"
    return {"id": rid, "source_id": rid, "split": split, "persona_id": "titan", "source": r["source"],
            "target": r["target"], "references": [r["target"]],
            **{k: r[k] for k in ("domain", "intent", "move", "length", "register")}, "origin": origin}


SPLITS = {"train": [row(r, "train", "titan_v1") for r in load("titan_train.jsonl")]
          + [row(r, "train", "handwritten") for r in load("handwritten_h301_h400.jsonl")],
          "validation": [row(r, "validation", "titan_v1") for r in load("titan_val.jsonl")],
          "test": [row(r, "test", "titan_v1") for r in load("titan_test.jsonl")]}
ids = [r["id"] for rows in SPLITS.values() for r in rows]
assert len(ids) == len(set(ids)), "duplicate ids"
sources = [r["source"].strip().lower() for rows in SPLITS.values() for r in rows]
assert len(sources) == len(set(sources)), "duplicate sources across splits"
for split, rows in SPLITS.items():
    with open(HERE / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
manifest = {"dataset": "titan_v1",
            "personas": {"titan": {"name": "Titan", "description": (
                "Calm, immense, patient; a being who has already decided. Short declaratives, no contractions, no "
                "exclamation marks, at most 60 words; every fact of the source kept (times, names, numbers, the "
                "request); one of ten rhetorical moves per line. Original writing only, no film dialogue.")}},
            "files": {s: len(r) for s, r in SPLITS.items()}}
json.dump(manifest, open(HERE / "manifest.json", "w"), indent=1)
print(manifest["files"])
