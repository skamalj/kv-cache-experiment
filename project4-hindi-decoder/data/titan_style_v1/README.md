# titan_style_v1

Style signals for the `titan_v1` data (`../titan_v1/train.jsonl`, `validation.jsonl`, `test.jsonl`; 1,260 / 100 / 100 rows).

## Part A — style attribute scores

**Files**

- `attributes.jsonl` — 1,460 lines, one per `titan_v1` row, in file order (train, then validation, then test). Each line: `id`, `split`, `source_style`, `target_style`; the two style objects carry the seven attributes `formality`, `grandiosity`, `dramatic_tone`, `philosophical_tone`, `authoritative_tone`, `elevated_vocabulary`, `aggression`, each 0.0–1.0 in steps of 0.1.
- `attributes_rescore_validation.jsonl` — 100 lines, the validation **targets** scored a second time, blind to the first pass; `target_style` only.
- `scoring_protocol.md` — the anchors and worked examples every score was taken from.
- `check_style_v1.py` — the checks below (`scores` and `rescore` sub-commands).

**How it was scored**

Every source and every target (2,920 sentences) was scored by reading the sentence on its own against the written anchors in `scoring_protocol.md`: for each attribute, pick the closest anchor, move one 0.1 step only if the sentence sits clearly between two anchors. Sources were scored as plain sentences someone said, targets as sentences someone said, without reference to the other half of the pair. No script, lexicon or heuristic assigned any score.

The work was split into four slices (train rows 1–420, 421–840, 841–1,260 including the handwritten `h301`–`h400`, and validation+test), each scored by a separate scorer instance (Claude Fable 5.1) working from the same protocol. Slice-level means agree closely (target formality 0.80 in all four; target grandiosity 0.50–0.52; target authoritative_tone 0.86–0.89), so the protocol rather than the scorer is doing the calibration.

**Consistency check** — the 100 validation targets were rescored by a fifth, independent instance that did not see the first pass. Mean absolute difference per attribute (aim ≤ 0.15):

| attribute | MAD | max diff | within 0.1 |
|---|---|---|---|
| formality | 0.033 | 0.1 | 100% |
| grandiosity | 0.051 | 0.1 | 100% |
| dramatic_tone | 0.059 | 0.2 | 98% |
| philosophical_tone | 0.045 | 0.2 | 99% |
| authoritative_tone | 0.039 | 0.1 | 100% |
| elevated_vocabulary | 0.016 | 0.1 | 100% |
| aggression | 0.017 | 0.2 | 99% |

Because the rescore was done by a different instance, this is inter-scorer agreement, the stricter version of the check.

**Separation (all 1,460 rows)**

| attribute | source mean | target mean | target > source |
|---|---|---|---|
| formality | 0.43 | 0.80 | 86% |
| grandiosity | 0.02 | 0.51 | 100% |
| dramatic_tone | 0.08 | 0.64 | 100% |
| philosophical_tone | 0.01 | 0.36 | 85% |
| authoritative_tone | 0.37 | 0.87 | 100% |
| elevated_vocabulary | 0.03 | 0.25 | 92% |
| aggression | 0.05 | 0.18 | 58% |

Notes on reading these: formality is not higher for every target because formal-notice sources already sit at 0.8–1.0. Aggression ties at 0.0 for roughly 40% of pairs, because most Titan lines carry no pressure and neither do their sources; the attribute separates only on warnings, complaints and threats, which is the behaviour the style head should learn. Philosophical_tone varies most across targets (0.0 for staccato instructions, 0.9 for maxim-built lines), so it is the attribute with the widest usable range on the Titan side.

**Checks** (run from this folder, `../titan_v1` alongside):

```
python3 check_style_v1.py scores  --ref <titan_v1 rows in order> --scores attributes.jsonl
python3 check_style_v1.py rescore --first attributes.jsonl --second attributes_rescore_validation.jsonl
```

Both report 0 problems on this delivery; the rescore reports PASS.

## Part B — contrast versions

**Files**

- `contrast.jsonl` — 600 lines: `id`, `split`, `move`, `formal`, `near_miss`. Order: the 100 validation rows, the 100 test rows, then the 400 train rows sorted by move (ledger, inevitability, correction, wisdom, threat, mission, mercy, limits, judgment, patience) and id.
- `contrast_brief.md` — the definitions, rules and examples the versions were written from.
- `check_contrast.py` — the checks below; it imports the fact rules from `../titan_v1/raw/validate_titan.py`.

**How the 400 train rows were chosen**

From train rows whose id starts with `titan_` (the handwritten `h301`–`h400` were excluded), 40 per move, picked by round-robin over domains within each move (seed 7, domains visited from most to least populated, one row per domain per pass) so that every move's 40 span 18–19 of the 20 domains and all 20 domains appear overall.

**How the versions were written**

Each version was written by hand per row (eight writer instances, Claude Fable 5.1, 75 rows each, from the same brief): `formal` as plain professional English with no device of the Titan register, `near_miss` as a stage-villain parody with the devices Titan forbids (exclamation marks, stacked cosmic nouns, archaic flourish, shouting). Nothing reuses the source or the Titan target verbatim; no line is duplicated across the 600.

**Fact-check result** — `python3 check_contrast.py --ref <the 600 reference rows> --out contrast.jsonl` reports 600 rows, 0 problems: every clock time, number, weekday, month and mid-sentence name of each source appears in both versions; no `formal` line has a contraction or an exclamation mark and all are 1–3 sentences; every `near_miss` is 1–5 sentences and at most 60 words; no version contains a blocklisted fragment. Length: `formal` 5–35 words (mean 17), `near_miss` 20–50 words (mean 35). All 600 `near_miss` lines carry at least one exclamation mark, which is the intended separation from Titan (which has none).
