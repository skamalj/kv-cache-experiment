# titan_v1: the Titan register

An everyday English sentence → the same facts in the **Titan register**:
- calm, immense and patient; short declaratives;
- no contractions, no exclamation marks; one to five sentences, at most 60 words;
- one of ten rhetorical moves per line;
- every time, name, number and request kept.

The register is original writing: no film dialogue, and the well-known lines are blocklisted (`raw/titan_style_guide.md`).

| Split | Rows | Notes |
|---|---|---|
| train | 1,260 | 1,160 delivered + 100 handwritten (h301–h400) |
| validation | 100 | early stopping |
| test | 100 | held out for the SFT vs SFT + JEPA comparison |

**Delivered data** (`raw/`):
- 1,360 hand-written pairs: ids 1–240, 401–1200 and 1361–1680, so 1,360 of the 2,000 planned in `raw/titan_dataset_plan.md`.
- Tags on every pair:
  - `domain` (20)
  - `intent` (12)
  - `move` (10): ledger, inevitability, correction, wisdom, threat, mission, mercy, limits, judgment, patience
  - `length` (short / medium / long, by source words)
  - `register` (casual / neutral / formal source)
- Splits by `raw/make_splits.py`, stratified by domain × move, seed 13.
- Checks: `raw/validate_titan.py`. These cover the style rules, fact preservation (numbers, clock times, weekdays, months, names), duplicates, repeated 4-grams and coverage against the plan.

**Handwritten h301–h400** (`raw/handwritten_h301_h400.jsonl`), train only:
- in the same register;
- 10 per move, 5 per domain;
- sources: 24 short, 43 medium, 33 long;
- all pass `validate_titan.py`;
- no source close to a validation, test or demo sentence (word overlap < 0.5) or to a delivered training source (< 0.6).

**Not merged from `thanos_v9/`.** Its targets are a different voice: close paraphrases, median 10 words, "…" and "shall". The strictest filter tried kept 61 rows: they pass the Titan validator, have at least 14 words and 2 sentences, contain no "…" or "shall", are not questions, and have a source of at least 7 words. Even these are about half the Titan length and carry no rhetorical move, so they would pull the register shorter and plainer.

**Build:** `python3 -I build_titan_v1.py`. It writes `train` / `validation` / `test` in the project's schema (`id`, `source_id`, `split`, `persona_id: titan`, `source`, `target`, `references`, plus the tags and `origin`) and `manifest.json`. The validator passes on the output: 0 rows flagged, no near-duplicate sources or targets.
