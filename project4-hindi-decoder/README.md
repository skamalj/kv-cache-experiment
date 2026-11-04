# Project 4 — Persona-language translation with a stitched encoder–decoder

Translate an English sentence into natural Roman-script Hinglish in one of six persona voices (Vijay of *Deewar*, Don, Shahenshah, Anthony Gonsalves, Vijay Dinanath Chauhan of *Agneepath*, Jai of *Sholay*), preserving its meaning, with a bidirectional encoder connected to a decoder through a trainable bridge and cross-attention.

## Start here

| | |
|---|---|
| Specification | [`docs/encoder_decoder_bridge_e2e_plan_v2.md`](docs/encoder_decoder_bridge_e2e_plan_v2.md) — plan v2, the governing document |
| Implementation plan | [`docs/stitched-e2e-plan.md`](docs/stitched-e2e-plan.md) — how v2 is carried out here, decisions, gates |
| Titan data | [`data/titan_v1/`](data/titan_v1/) — English → the **Titan register** (calm, immense, patient; short declaratives, one of ten rhetorical moves, every fact kept): 1,360 delivered pairs + 100 handwritten; train 1,260 / validation 100 / test 100 (test held out for the JEPA A/B); own style guide and validator in `raw/` |
| Thanos data | [`data/thanos_v9/`](data/thanos_v9/) — English → Thanos-voice English in **the user's own voice**: train 2,145 rows (1,665 unique; the user's 480 own lines weighted ×2) / validation_daily 24 / validation 25 (reference) / test 25. Earlier: [`thanos_v8/`](data/thanos_v8/) (one dramatic voice, run R15), [`thanos_v7/`](data/thanos_v7/) (same inputs, mixed voices), [`thanos_v6/`](data/thanos_v6/) (200 handwritten; split run), [`thanos_v5/`](data/thanos_v5/) (100 handwritten, run R13), [`thanos_v4/`](data/thanos_v4/) (run R12), [`thanos_v3/`](data/thanos_v3/) (half Roman-Hinglish inputs), v2 |
| Ledger | [`LEDGER.md`](LEDGER.md) — integration notes (blog material) and run entries |
| Original brief | [`docs/Project 4 Requirements.md`](docs/Project%204%20Requirements.md) — where it differs, plan v2 governs |
| Data | [`data/persona_translation_v3/`](data/persona_translation_v3/) — 60 English sources × 6 personas, split by source 240 / 60 / 60; **v4** (three personas): [`data/persona_translation_v4/`](data/persona_translation_v4/) — two deliveries merged; train 768 pairs, validation 75, test 75, out-of-domain 114, minimal pairs 72 ([spec](docs/data-requirements-v4.md)) |

## Stages

| Folder | What | Status |
|---|---|---|
| [`1-decoder-comparison/`](1-decoder-comparison/) | 8 decoder variants × 40 cases for Hindi / Hinglish generation; requirement, notebook, prompts, results, scoring sheet, draft report | done; rubric scores pending |
| [`2-hinglish-two-models/`](2-hinglish-two-models/) | Qwen2.5-1.5B + Hinglish adapter vs Gemma 3 1B IT, two passes | done; Gemma chosen at the time (Qwen never stopped) |
| [`3-persona-pilots-superseded/`](3-persona-pilots-superseded/) | persona LoRA on the earlier *scenario → reaction* task: Gemma 3 1B / 4B (runs 1–4, in git history), Qwen base + adapter | **superseded** by the translation task; kept for the findings (stop-token training, 4-bit Gemma, untrained `<|im_end|>` in Qwen base) |
| [`4-translation-baseline/`](4-translation-baseline/) | **Phase 1**: direct-source decoder baseline on data v3, one notebook per decoder (Gemma 3 4B IT, Qwen2.5-1.5B-Instruct, Qwen-Instruct + Hinglish adapter as diagnostic); per-persona scores | done; Gemma 3 4B IT chosen; Don, Anthony, Jai kept |
| [`5-bridge-skeleton/`](5-bridge-skeleton/) | **Phase 2**: ModernBERT-base + bridge (GPU 0) → gated cross-attention into Gemma 3 4B IT (GPU 1); plumbing tests, content-flow test, smoke runs (3 runs in git history) | done: plumbing proven; content flows once shortcuts are closed (no biases, random-word copy); meaning not learned from 120 pairs with frozen LoRAs |
| [`6-bridge-training/`](6-bridge-training/) | **Phase 3**: copy warm-up, then encoder LoRA + bridge + cross-attention + decoder LoRA trained on v4 (768 pairs, Gemma 3 4B decoder sees the persona instruction) | done: trains cleanly, but the output does not follow the sentence (a few training topics recur; chrF 15.7) |
| [`7-simple-bridge/`](7-simple-bridge/) | **Simplified architecture**: persona + sentence → ModernBERT + LoRA → bridge → Gemma 3 4B IT (4-bit) + LoRA; the decoder gets nothing but the bridge (its only input is `<bos>`); all trained together on v4, leaving two LoRAs | done: collapses to one stock line per persona (5 distinct of 75; chrF 12.0): the bridge carries the persona, not the sentence |
| [`8-bridge-pretraining/`](8-bridge-pretraining/) | **Same architecture, pre-trained first**: stage 1 plain English → Hinglish on 24,000 public pairs (Hinglish-TOP, Apache-2.0; PHINC, CC BY 4.0), stage 2 persona fine-tuning on v4 | done: no more collapse (validation 71 of 75 distinct, chrF 18.4; test 17.8; out-of-domain 18.4); the bridge carries the kind of sentence (a weather query, a time and a place, money) but not its specifics or meaning |
| [`9-prefix-bridge/`](9-prefix-bridge/) | **Prefix design**: the encoder's output, through an MLP bridge, is fed once as input vectors after `<bos>`; Gemma + LoRA reads it with its own attention (no cross-attention blocks); same two stages (stage 1 default 12,000 pairs, balanced) | **done, best so far:** stage 1 held-out chrF 41.9 (vs 27.6); v4 validation chrF 31.9, test 29.6, out-of-domain 33.3, with names, numbers and meaning mostly carried. Thanos run (R10): stage 1 English reconstruction through the bridge reaches chrF 98.8 (sentences carried almost intact); stage 2 on thanos_v3 keeps meaning for English inputs but the voice is weak, and Roman-Hinglish inputs fail. **Now Thanos-only, English only (data thanos_v4, 5× the training pairs):** R12 validation chrF 36.1 (from 27.7), test 60.3; meaning kept, voice still light; a 6,000-sentence stage 1 failed (R11), so 12,000 stays. Final cell translates six chosen sentences |
| [`10-prefix-bridge-split/`](10-prefix-bridge-split/) | **Prefix design, split roles**: same model as `9-prefix-bridge/`; R14 (Gemma frozen in stage 1, data thanos_v6) never learned to carry the sentence, so stage 1 now trains Gemma's LoRA too (`S1_DECODER_LORA = True`) and resets it before stage 2; data thanos_v8, early stopping on validation_daily, but stage 1 trains only the encoder LoRA and the bridge (Gemma frozen) and stage 2 trains only a fresh Gemma LoRA (encoder and bridge frozen), so the LoRA that learns the voice never learned to copy; stage 1 prints chrF every 500 steps; fallback `S1_DECODER_LORA` | ready to run |
| [`11-prefix-bridge-two-paths/`](11-prefix-bridge-two-paths/) | **Carry vs reset in one run**: one stage 1, then stage 2 twice from the same weights (carry: the stage-1 Gemma LoRA continues; reset: a fresh one), each scored on all held-out sets, the chosen sentences and eight sampled variants, then side by side; data thanos_v9 | ready to run |
| [`12-jepa-control/`](12-jepa-control/) | **JEPA control, data titan_v1**: R16's reset path plus a 6-layer transformer predictor that writes K = 8 latent slots after the x prefix. Two arms, identical except the loss: **control** (cross-entropy only) vs **jepa** (+ λ · (1 − cos) to the frozen stage-1 latent of the Titan sentence, pooled into 8 slots). Stage 1 reused from `/kaggle/working/` if present, else trained here. Scored on chrF, facts kept, style, and a slot-swap test. Bar: +2 chrF on test, no loss of faithfulness | ready to run |

`data/persona_pilot_60.json` and `data/persona_pilot_168.json` belong to the superseded pilots; they stay at their paths because those notebooks load them from GitHub.

## Conventions

- Notebooks run on Kaggle (T4). The Gemma notebooks need a Hugging Face token in Cell 0; clear it before saving.
- Each notebook prints its full results and records into its own outputs; the saved notebook is the result.
- Data is loaded from this branch on GitHub, so data changes must be pushed before a run.
