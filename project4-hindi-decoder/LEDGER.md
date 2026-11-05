# Project 4 — Ledger

Two kinds of entries, newest last in each section:

- **Integration notes**: problems met while wiring the pieces together, why they happen, and the fix. Written to be lifted into the blog.
- **Runs**: one entry per Kaggle run (to be filled from the notebooks' saved outputs).

## Integration notes

### I1 — Generating from a prefix of embeddings with Gemma 3 (2026-10-20, `9-prefix-bridge/`, commit `57793db`)

**Setting.** The prefix bridge feeds the encoder's output to Gemma 3 4B IT as input vectors: `[<bos>] + prefix vectors + target`, passed as `inputs_embeds`. Training works: a plain forward pass of `Gemma3ForConditionalGeneration` accepts `inputs_embeds`.

**Problem.** Generation does not:

```
ValueError: You passed `inputs_embeds` to `.generate()`, but the model class Gemma3ForConditionalGeneration
doesn't have its forwarding implemented.
```

**Why.** `generate()` only accepts embeddings as the starting point for model classes whose generation-input preparation supports it (GPT-2 and most text-only decoders do). Gemma 3's multimodal class (`transformers` 5.17) does not, because it builds its own inputs from token ids (it looks for image tokens in them).

**Fix.** Give `generate()` token ids it can work with and swap in the vectors underneath:

1. Build the prompt as ordinary ids: `<bos>` followed by one **placeholder token** per prefix vector (any fixed token; we use `" ."`).
2. Register a **forward hook on Gemma's input-embedding layer**. When a prefix is waiting and the embedding output has the prompt's length (`1 + number of prefix vectors`), the hook replaces positions `1 …` with the prefix vectors and clears the waiting prefix.
3. That happens on the first (full-prompt) forward pass only; every later step embeds a single new token, so the hook leaves it alone, and `generate()` runs normally with its KV cache.

```python
def write_prefix(module, args, output):
    if SLOT.vectors is not None and output.dim() == 3 and output.shape[1] == 1 + len(SLOT.vectors):
        output = output.clone()
        output[:, 1:] = SLOT.vectors.to(output.dtype)
        SLOT.vectors = None
    return output

EMBED.register_forward_hook(write_prefix)          # EMBED = dec_model.get_input_embeddings()
```

**Guard.** `translate()` asserts the prefix was consumed, so a silent failure (the hook not firing, wrong length) stops with an error instead of producing text from placeholder tokens. Training is unaffected: it passes `inputs_embeds` and never calls the embedding layer with a waiting prefix.

**Why it is a correct substitute.** Gemma's embedding layer applies its own scaling (`× √hidden`); the hook replaces the *output* of that layer, so the prefix vectors enter at the same point and scale as in training, where they were concatenated with already-scaled token embeddings.

### I2 — Long runs on Kaggle: a silent kernel death, and pinned versions that did not install (2026-10-20, `9-prefix-bridge/`)

**What happened.** Stage 1 of the prefix bridge died after step 1000 of 1500, about 70 minutes in, with no Python error: the kernel simply restarted. GPU memory was not the cause (peak 8.1 GB of 15; a GPU shortage raises `OutOfMemoryError` instead). A kernel death without an error is almost always the machine's RAM. All progress was lost, because nothing was saved until the end of the stage.

The same run also printed `transformers 5.16.1, peft 0.20.0` although Cell 1 installs `5.17.0` and `0.21.2`, with warnings `Ignoring invalid distribution ~ransformers`. An earlier interrupted install had left a half-removed package in the environment, and the pinned install silently did not take.

**Fixes.**

- **Checkpoint and resume.** Every 500 steps, Cell 6 saves weights, optimizer state, scheduler state, step and log to `/kaggle/working/stage1_ckpt.pt`. On a restarted kernel, running all cells again rebuilds the same corpus (fixed seed) and the cell resumes from the checkpoint. It checks that the checkpoint belongs to a corpus of the same size.
- **Less RAM.** The full corpus tables (≈ 190k rows) are deleted once the 12,000 pairs are sampled, and `gc.collect()` plus `torch.cuda.empty_cache()` run at every checkpoint.
- **Pinned versions enforced.** `pip install --force-reinstall --no-deps` for the two pinned packages, and an assertion that stops the notebook if the versions in use are not the pinned ones.

**Lesson for the blog.** On a shared notebook service, a multi-hour training loop needs checkpoints as much as a cluster job does. Also, *check* the versions you pinned; don't assume the install worked.

## Runs

*Earlier runs (Phases 1–3, notebook 7) are summarised in the README's stage table and in their commit messages; full entries to be backfilled.*

### R8 — Cross-attention bridge, pre-trained (2026-10-20, `8-bridge-pretraining/`, commit `52f21f7`)

- **Setup:** ModernBERT-base + LoRA → bridge (Linear 768→2560) → 8 gated cross-attention blocks (≈ 210 M, from scratch) in Gemma 3 4B IT (4-bit) + LoRA; the decoder's only input is `<bos>`. Stage 1: 24,000 public English → Hinglish pairs (Hinglish-TOP + PHINC), one pass (3,000 steps of 8). Stage 2: v4 persona data (768 pairs), kept epoch 1 of 3.
- **Stage 1:** held-out loss 9.43 → 3.08; on 120 held-out corpus pairs chrF **27.6**, 119 distinct. Output keeps the kind of sentence and its grammar but swaps specifics ("weather in Kingston" → "New York me mausam kaisa hai").
- **Stage 2 (v4):** chrF validation **18.4** / test 17.8 / out-of-domain 18.4; 71 of 75 distinct (no collapse, against 5 of 75 without pre-training). Meaning mostly not preserved ("Kabir lands in Delhi at 9 tonight" → "Aaj raat 11 baje Mumbai ka traffic kharaab hoga").
- **Decision:** the cross-attention design needs far more data to carry details; try the prefix design (R9).

### R9 — Prefix bridge, pre-trained (2026-10-20, `9-prefix-bridge/`)

- **Setup:** ModernBERT-base + LoRA → bridge (MLP 768→2560→2560 + RMSNorm at Gemma's embedding scale, ≈ 8.5 M) → one input vector per English token, placed after `<bos>`; Gemma 3 4B IT (4-bit) + LoRA reads them with its own attention. No cross-attention blocks. Stage 1: **12,000** public pairs (6,000 Hinglish-TOP + 6,000 PHINC), 1,500 steps of 8 (one kernel death at ≈ step 1000; resumed from checkpoint, see I2). Stage 2: v4, kept epoch 2 of 4. Integration fix for generation: I1.
- **Stage 1:** held-out loss 9.57 → **2.28** (3.36 at step 1000); chrF **41.9**, 120 of 120 distinct. Specifics now carried ("weather in Kingston" → "Kingston me mausam kaisa hai"; "Berry Hill" → "Bar Hill"; "tomorrow" → "kal").
- **Stage 2 (v4):** chrF validation **31.9** / test **29.6** / out-of-domain **33.3**; 75 of 75 distinct; stopping 99–100 %. Meaning largely preserved, persona voices present: "Kabir lands in Delhi at 9 tonight" → "Kabir aaj raat Delhi pahunch jaayega" (the time dropped); "The plumber charged 4,000 for 2 hours…" → "Plumber ne 4,000 paise 2 ghante mein lete hain. Kya woh rate hai jo usne di?"; Anthony: "AC phir se band ho gaya—yeh bhi reboot karna hoga!".
- **Comparison:** with half the stage-1 data and ≈ 4 % of the new parameters, the prefix design beats the cross-attention design on every measure (stage-1 chrF 41.9 vs 27.6; v4 validation 31.9 vs 18.4).
- **Open:** the direct-source baseline on v4 (Gemma reading the English in its prompt) to judge how much the bridge still loses; occasional dropped numbers or details.

### R10 — Prefix bridge, Thanos target (2026-10-21, `9-prefix-bridge/`, `TARGET = "thanos"`, data `thanos_v3`)

- **Setup:** as R9. Stage 1 = **English reconstruction** ("Plain: X" → X) on 12,000 English sentences (Hinglish-TOP + PHINC English sides), 1,500 steps of 8. Stage 2 = thanos_v3 (100 pairs; half English, half Roman-Hinglish inputs), kept epoch 2 of 4.
- **Stage 1 — the stitching proof:** held-out loss 8.13 → **0.058**; on 120 held-out sentences chrF **98.8**, 120 of 120 distinct; essentially every sentence comes through the encoder → bridge → Gemma path intact, names and numbers included (one near miss: "Pham" → "Pam").
- **Stage 2 (Thanos):** chrF validation **27.7** / test **38.0**; stopped 100 %; 25 of 25 distinct; copies of the source 3 / 25 (validation), 1 / 25 (test).
  - **English inputs:** meaning kept, voice weak — mostly light paraphrases ("We must celebrate the progress made, not dismiss the work that remains."); chrF 32.4 / 51.0.
  - **Roman-Hinglish inputs:** meaning lost, fluent invented English ("Pehli koshish fail ho…" → "The promise of the harvest is broken."); chrF 22.6 / 26.0. The encoder (ModernBERT, English) and stage 1 (English only) never learned to read Roman Hinglish.
- **Conclusions:** (1) the bridge carries an English sentence into Gemma almost perfectly — the core stitching result; (2) 100 Thanos pairs after a copy-heavy stage 1 give faithful but under-styled output; (3) Roman-Hinglish inputs need an encoder that reads them (multilingual encoder, or Hinglish in stage 1) or should be excluded (`THANOS_INPUTS = "english"`).


### R11 — Prefix bridge, Thanos v4, stage 1 cut to 6,000 sentences (2026-10-22, `9-prefix-bridge/`, stopped in stage 2)

- **Setup:** as R10, but English only (data `thanos_v4`: train 498, validation 25, test 25) and stage 1 shortened to **6,000** sentences (750 steps of 8), hoping a shorter copy stage would leave Gemma less locked into copying.
- **Stage 1:** held-out loss 8.13 → 3.21 (step 500) → **2.99** (step 750); chrF **23.1** on 120 held-out sentences. The bridge carried only the kind of sentence: every place became "New York", "Send a video message to Pham" → "Send a message to my mom", "Should I wear shorts tomorrow?" → "Should I bring an umbrella?".
- **Stage 2 (stopped):** validation loss 4.46 at epoch 0; outputs unrelated to the input.
- **Lesson:** reconstruction does not improve gradually; it *clicks*, between steps 500 and 1,000 in R10 (held-out loss 3.15 → 0.11). With 750 steps and a learning rate decaying to zero, the run ended before the click. Stage 1 needs at least about 1,000 steps at this learning rate.

### R12 — Prefix bridge, Thanos v4, stage 1 12,000 sentences (2026-10-23, `9-prefix-bridge/`)

- **Setup:** as R11 with `CORPUS_TOTAL = 12000` (1,500 steps). All three parts (encoder LoRA, bridge, Gemma LoRA) trained in both stages.
- **Stage 1:** identical to R10 (same seed and corpus): held-out loss 3.15 → 0.11 → **0.058**; chrF **98.8**, 120 of 120 distinct. 1,500 steps in 5,547 s (3.7 s per step).
- **Stage 2:** validation loss 6.19 → **2.06** (epoch 1) → 2.14 → 2.51; early stop, kept epoch 1. Overfits fast: train loss 1.32 → 0.33 by epoch 3. About 3.5 min per epoch.
- **Scores:** validation chrF **36.1** (R10: 27.7), test **60.3** (R10: 38.0); stopped 100 %; 25 of 25 distinct; copies of the source 1 / 25 and 0 / 25; no copies of training targets.
  - The test score is high partly because the v3 test targets stay close to the source's wording; the curated validation targets add a second, interpretive sentence (13.5 words on average against outputs of 10.1), which chrF cannot reward.
- **What the output looks like:** meaning kept, register raised, but light Thanos: formal markers ("Should the engine make that noise again…", "yet", "Let us", "Grandmother", "is required") rather than the gravity and second sentence of the references. Lowest validation scores are mostly fair paraphrases ("Don't confuse being busy with making progress." → "Do not mistake effort for progress.", chrF 15), with two real errors: "The signal is weak" → "The alarm is weak"; "a fair outcome" → "a firm foundation".
- **Six demo sentences (Cell 10):** all meanings kept, all fluent; voice light. Two shifts: "HR has asked me to submit my appraisal…" → "My assessment must be submitted in two days. It covers my entire year's work." (HR dropped); "I don't think we should sign the lease until…" → "The lease will not be signed until the heating is fixed." (the hedge became a decision).
- **Conclusions:** (1) 5× the training pairs and English-only inputs lift validation chrF from 27.7 to 36.1 with no loss of meaning; (2) the voice is still weak, and stage 2 peaks after one epoch: Gemma's LoRA, trained to copy in stage 1, moves only a little towards the style before it starts memorising. **Next:** split the roles (stage 1 trains encoder LoRA + bridge only, Gemma frozen; stage 2 trains a fresh Gemma LoRA only), so the copy habit never enters the LoRA that learns the voice.

### R13 — Prefix bridge, Thanos v5 "data only" (2026-10-23, `9-prefix-bridge/`)

- **Setup:** as R12; only the data changes: `thanos_v5` = v4 + 100 handwritten pairs in a stricter, weightier voice (train 598). New final-cell sentences, kept out of training.
- **Stage 1:** identical to R10/R12 (chrF 98.8).
- **Stage 2:** validation loss 6.19 → **2.13** (epoch 1) → 2.36 → 2.65; early stop, kept epoch 1 again. About 4.4 min per epoch.
- **Scores:** validation chrF **36.3** (R12: 36.1), test **61.5** (R12: 60.3); stopped 100 %; 25 of 25 distinct; copies of the source 0 / 25 and 2 / 25.
- **Voice:** slightly bolder on the final-cell sentences, which now show the restructured two-sentence form: "I worked all weekend to fix the bug, and nobody noticed." → "All weekend, I worked to fix the bug. No one noticed."; "The project is three weeks late, but the client still wants it by Friday." → "The project is three weeks late. The client still demands it by Friday."; "I trained for a year, and I missed the medal by one second." → "For a year, I trained. One second, and the goal was lost." On validation and test, still mostly light rewording (4 of 25 outputs have two sentences, as in R12).
- **Faithfulness errors** (some shared with R12, some new):
  - numbers: "18,000 … 22,000" → "eight hundred … two thousand" (R12: "eighteen thousand … twenty thousand"); "2:20 … 2:05" → "two-forty … two-fifty" (R12: also wrong). Converting digits to words through the bridge is unreliable;
  - names and spelling: "Sana" → "Sano" (R12: "Sanna"); "crates" → "crats" (both runs);
  - meaning, new in R13: "longer" → "shorter"; "I will stand by you, but I won't lie for you." → "Stand by me, and I will not lie for you."; "If the information is incomplete…" → "If the matter is weak, speak plainly."
- **Conclusions:** 100 strict, weighty pairs (17 % of the train set) do not move the overall voice measurably; stage 2 still peaks after one epoch, which points at the training set-up rather than the data. Next: `10-prefix-bridge-split/` on the same data (Gemma frozen in stage 1, a fresh Gemma LoRA alone in stage 2). Open: digits in inputs (all targets write numbers as words), worth a dedicated check.

### R14 — Split roles, Gemma frozen in stage 1 (2026-10-23, `10-prefix-bridge-split/`, stopped at step 1,000)

- **Setup:** as R13 but data `thanos_v6` (train 698) and **split roles**: stage 1 trains only the encoder LoRA and the bridge, Gemma frozen (its LoRA at B = 0); stage 2 was to train a fresh Gemma LoRA only. Stage 1: 12,000 sentences, 1,500 steps planned.
- **Stage 1:**

  | Step | held-out loss | chrF (40 held-out) | R10 held-out loss |
  |---|---|---|---|
  | 0 | 8.13 | — | 8.13 |
  | 500 | 3.45 | 20.6 | 3.15 |
  | 1,000 | **3.06** | **26.7** | **0.11** |

  Outputs at step 1,000 keep the kind of sentence but not its content ("Texting Pascal, don't forget the flowers." → "Tell mom I'm done with the chores."). Time and memory as before (3,782 s to step 1,000; peak 8.7 GB): the gradient still passes through Gemma.
- **Stopped** at step 1,000: with the learning rate decaying to zero over the last 500 steps, the click would not come, and stage 2 on a bridge that does not carry the sentence is wasted (R11).
- **Lesson:** at these learning rates (bridge 3e-4, encoder LoRA 1e-4) and this step budget, the bridge alone cannot steer a frozen Gemma into repeating a sentence; Gemma's LoRA has to meet it halfway. The sentence-carrying click needs both sides trained.
- **Next:** the fallback `S1_DECODER_LORA = True`: stage 1 trains all three parts (as R10, clicks by step 1,000), then Gemma's LoRA is **reset to zero** and stage 2 trains a fresh one alone, encoder and bridge frozen. This still tests the hypothesis that the LoRA learning the voice should never have learned to copy.

### R15 — Split roles with reset, dramatic data (2026-10-23, `10-prefix-bridge-split/`, data `thanos_v8`)

- **Setup:** `S1_DECODER_LORA = True`: stage 1 trains encoder LoRA, bridge and Gemma LoRA (12,000 sentences, reconstruction); then **Gemma's LoRA is reset to zero** and stage 2 trains a fresh Gemma LoRA **alone** (encoder and bridge frozen). Data `thanos_v8` (train 1,157, one dramatic voice); early stopping on `validation_daily` (the user's everyday set, 24).
- **Stage 1:** held-out loss 3.08 (500) → 0.122 (1,000) → **0.060** (1,500); chrF on 40 held-out 24.5 → 97.1 → 98.3; final chrF **98.7** on 120. 7,327 s (slower than R10 by the per-checkpoint chrF and the session).
- **The reset is safe.** At stage 2 epoch 0, after the reset and before any stage-2 training, plain Gemma still reads the bridge: "I'll grab lunch and be back in twenty minutes." → the same sentence; "I'm running late because I couldn't find parking." → "I'm going to be late because I can't find parking." The bridge's vectors, learned together with the stage-1 LoRA, are readable by Gemma without it.
- **Stage 2:** validation_daily loss 2.83 → **1.11** (epoch 1) → 1.13 → 1.46; early stop, kept epoch 1. About 10 min per epoch.
  - epoch 1: "Parking was beyond my reach, and so I ran late."; "Twenty minutes I shall spend gathering lunch, and then I shall return."; "Will you stand guard while I step out?"
  - epoch 3 drifts: "Will you stand ready to bear witness to my departure?" for "Can you cover for me while I step out?"
- **Scores:** validation_daily chrF **48.3**; validation (old curated set, reference) 36.2 (R13: 36.3); test 56.1 (test targets now dramatic, so not comparable with R13's 61.5); stopped 100 %; all outputs distinct; copies of the source 0 / 4 / 0.
- **Voice:** clearly dramatic for the first time, in one epoch: "All weekend I toiled to fix the bug, and no one noticed."; "Three weeks the project has been delayed, yet the client still demands it by Friday."; "Your lie did not anger me. It was your belief in your own necessity that I find disappointing."
- **The user's manual test** (expected → got):
  - "The traffic delays my arrival. I shall reach the office… when the path permits." → "Traffic shall keep me late to the office."
  - "I have been delayed by ten minutes. Proceed without me… do not wait." → "Ten minutes late I run. Begin without me, if you would."
  - "Hunger has made its presence known. Let us seek sustenance." → "My hunger gnaws at me. Let us find something to eat."
  - "The day's labors have taken their toll. I require… rest." → "Today's work has left me exhausted. I require some rest."
  - "I remain bound by this meeting. When I am free… I shall contact you." → "I am trapped in a meeting. I shall call you when I am free."
  All faithful (ten minutes kept); voice present but plainer than the user's lines; no "…" pauses (rare in v8).
- **Errors:** "borrowed my car" → "**stole** my car" (meaning changed); "too many emails to answer" → "I must answer each one today"; occasional inverted word order ("Ten minutes late I run", "one second I lost the medal"); one copy ("The generator can run for three hours without refuelling.").
- **Conclusions:** (1) one consistent dramatic voice in the data plus a fresh stage-2 LoRA gives the voice that R12/R13 lacked; (2) resetting Gemma's LoRA does not cost the meaning link; (3) stage 2 still peaks at epoch 1 and drifts after, so less training (or a lower LR) is the next knob. Next: the same notebook with both paths after stage 1, **carry** (stage-1 LoRA kept) vs **reset**, side by side.
- **Sampling on R15** (added after the run, temperature 0.9, top-p 0.95, 8 samples of "I'll be late to the office because of traffic."; greedy gives "Traffic shall keep me late to the office."): two at the user's level ("The office will not see me early. Traffic has forced me to delay my arrival."; "The office awaits me, though traffic shall have its claim on my time."), two good, two plain, two wrong ("…by its last hours" adds a time; "Traffic has forced me to make the office late" is garbled). The model has learned the style; greedy decoding picks the commonest, plainer phrasing. Two ways forward: data where the user's moves are the norm (thanos_v9), and best-of-n sampling with a faithfulness check.

### R16 — Carry vs reset from one stage 1 (2026-10-24, `11-prefix-bridge-two-paths/`, data `thanos_v9`)

- **Setup:** one stage 1 (encoder LoRA + bridge + Gemma LoRA, 12,000 sentences); then stage 2 twice from the same stage-1 weights, Gemma's LoRA trained alone (encoder and bridge frozen): **carry** (the stage-1 Gemma LoRA continues) and **reset** (Gemma's LoRA zeroed first). Data `thanos_v9`: train 2,145 rows (1,665 unique; the user's 480 own lines ×2), the user's measured voice throughout. Early stopping on validation_daily.
- **Stage 1:** clicked early: held-out loss 0.57 and chrF 80.7 already at step 500 (R15: 3.08 and 24.5), then 0.072 / 98.3 (1,000), 0.052 / 98.8 (1,500); final chrF **98.5**. Same corpus, seed and prefix scale (1.029) as R15, so the earlier click comes from non-deterministic GPU arithmetic near a tipping point: **the timing of the copying click varies between identical runs by hundreds of steps.**
- **Epoch 0 (before stage-2 training):** carry copies the encoder input **including its tag** ("Thanos: I'm running late because…"); reset gives near-copies without the tag ("I'm running late because I couldn't find the parking lot.").
- **Stage 2:** both kept epoch 1; validation_daily loss carry 0.982, reset **0.968**; both drift after epoch 1 (epoch 3: 1.66 / 1.71). Epoch 2 is more vivid in both ("a parking spot was beyond my reach") but loses on loss.

| | carry | reset |
|---|---|---|
| validation_daily chrF | 53.5 | 52.8 |
| validation chrF (reference) | 37.9 | 37.0 |
| test chrF | 61.7 | **63.2** |
| copies of the source (val_daily / val / test) | 3 / 4 / 4 | **1** / 5 / 4 |

- **Faithfulness decides it:** carry "borrowed my car" → "has taken my car", "wants" → "demands", "late to the office" → "late at the office"; reset keeps "borrowed" and "wants". Reset's own slips: "Ten minutes behind me… I am running late"; "For the first time, someone must do it." **Reset is the better path** (confirms R15: resetting is safe and avoids the copy habit; carry is negative transfer of the copy behaviour).
- **Voice: plainer than R15.** v9 follows the user's measured 500-file voice and weights it ×2; outputs are more faithful than R15 (no "stole") but less dramatic ("I worked all weekend to fix the bug… and no one noticed." vs R15's "All weekend I toiled to fix the bug, and no one noticed."). The drama level comes from the data, not the method.
- **Decision:** R15 → **Thanos v1** (more dramatic). This run's reset path is a candidate for v2 **after** a more dramatic data pass. Since reset zeroes Gemma's LoRA and stage 2 freezes encoder and bridge, stage 2 needs only this run's `encoder-lora/` and `bridge.pt`: the next experiments can skip stage 1 (~2 h).

### R17a: JEPA control, stage 1 retrained here (2026-11-04, `12-jepa-control/`, data `titan_v1`, stopped)

- **Setup:** first run of notebook 12, at the time with two arms (control vs jepa). R16's `encoder-lora/` and `bridge.pt` were no longer in `/kaggle/working/`, so stage 1 trained here with the same recipe as notebook 11 (12,000 sentences, batches of 8). The only difference was that the predictor was created before the LoRAs, which shifts their random start.
- **Stage 1 did not click:** chrF on 40 held-out sentences was 17.0 at step 500 and 29.2 at step 1,000 (train loss 3.10). R16 had 80.7 and 98.3, R15 24.5 and 97.1, and R14 (never clicked) 20.6 and 26.7. Stopped at step 1,000.
- **Lesson:** the copying click is fragile and depends on the starting point, not just the recipe. Keep a known-good stage 1 instead of retraining it: R16's `encoder-lora/` and `bridge.pt` are now in the private repo `skamalj/titan-stage1` (`r16/`), and notebook 12 loads them from there.
- **Next:** notebook 12 extended to four arms (control → jepa → style → contrast) using `titan_style_v1`, with every checkpoint mirrored to that repo.

### R17b — JEPA ladder on the bridge (2026-11-05, `12-jepa-control/`, data `titan_v1` + `titan_style_v1`)

- **Setup:** R16's stage 1 (from the private repo `skamalj/titan-stage1`, `r16/`). A 6-layer predictor writes 8 content slots + 2 style slots after the x prefix. Four arms, each adding one loss: control (CE) → jepa (+ cosine to the frozen stage-1 latent of y) → style (+ style head on 7 attribute scores) → contrast (+ InfoNCE against formal / near-miss versions). Modal L40S, about 3.5 min per epoch (Kaggle T4: about 18 min).
- **First pass:** the predictor's JEPA cosine stayed at 0 (raw outputs standardised, learning rate 1e-4 could not reach the target's scale). Fix: the predictor writes in the standardised space, learning rate 3e-4, plus a warm-up on the latent losses alone. The cosine then reached 0.24 in seconds.
- **Results (test):** chrF 27.4 / 28.0 / 27.5 / 27.6. Facts within noise. Titan rate 97–99 % in every arm. Slot swap changed 6 % / **94 %** / 44 % / 4 % of outputs: Gemma reads ẑ only in the jepa arm, and the contrast loss produces sentence-independent slots.
- **Blind judgement (test, 400 outputs, mine):** usable 52 / 49 / 56 / 56 %, all differences inside the noise.
- **Oracle check:** the true latent s(y) in place of ẑ changed 72 % of outputs, but no better (chrF 27.8 against 28.3; blind 8 against 10). The bridge latent is not worth predicting: the meaning errors come from the x prefix and the encoder.

### R18 — LLM-JEPA on Gemma, plain LoRA vs LoRA + JEPA (2026-11-05, `13-llm-jepa/`)

- **Design:** no encoder or bridge; Gemma + LoRA with the prompt "Rewrite in the Titan's voice: x". JEPA term: middle-layer state of (prompt + 2 `[PRED]`) against the y view, standardised cosine, stop-gradient on y.
- **4B, 300 pairs:** plain LoRA test chrF 29.0 (the best on test so far), numbers 0.93, names 1.0. JEPA: chrF 28.0, blind 24 against 29 (p = 0.58). No gain.
- **4B zero-shot:** the register cannot be prompted. The titan prompt gives a mythic parody (style rules 8 %); the described style comes out robotic and invents facts ("14:37 hours"); the thanos prompt gives film clichés.
- **4B, 50 pairs:** degrades gently; negation 0.52. Blind 29 against 38 for 300 pairs (p = 0.33).
- **1B, 50 pairs (LoRA fails visibly):** loops, incoherence, invented facts; validation CE 2.88. LoRA + JEPA (tokens) won the blind judgement **16 to 2 (p ≈ 0.001)**, names 0.86 against 0.57, at JEPA cosine 0.03. With a 6-layer predictor: 16 against 17, worse facts, cosine 0.26.
- **Rule set (user):** chrF does not fit this style transfer; verdicts use facts + blind judgement only.

### R19 — Plain-LoRA hyperparameter search, then JEPA on the best recipe (2026-11-05, `14-lora-sweep/`, 1B, 50 pairs)

- **Sweep:** 21 configurations (lr {5e-5, 1e-4, 2e-4} × rank {8, 16, 32} × dropout {0.05, 0.15}, then NEFTune × weight decay), up to 30 epochs, early stopping with patience 3. Best validation CE 2.8779 (lr 1e-4, r 16, dropout 0.15, weight decay 0.05). The **whole grid spans 2.878–2.907**, and the default recipe was 5th (2.8795). Cost about $0.55.
- **JEPA on the tuned recipe** (tokens and 6-layer, λ 0.5 / 1 / 2): facts equal; blind on validation against the tuned LoRA 18 to 16 (tokens) and 16 to 13 (6-layer), both null. **R18's 16-to-2 win was a regularisation effect** that dropout + weight decay also deliver.
- **JEPA warm-up (JEPA alone first):** collapsed (validation cosine 0.98; the model got worse, CE 3.01). The target was computed by the same LoRA model.

### R20 — The LLM-JEPA paper's claims, paper recipe (2026-11-05, `15-llm-jepa-claims/`, 1B)

- **Recipe as in Huang, LeCun and Balestriero (arXiv 2509.14252):** last token of the final layer, k = 1 `[PRED]`, plain cosine, gradient through both views, CE + λ·JEPA from step one. Tuned LoRA recipe. 20 epochs with no early stopping; general-ability check = NLL on the base model's own answers to 50 general questions.
- **50 pairs:** the best validation CE is plain LoRA's (2.879 at epoch 5; JEPA λ 1/2/4: 2.880 / 2.897 / 2.914). **No faster convergence.** After the peak JEPA overfits **more slowly**, and more so at higher λ (epoch 10: 3.39 at λ 4 against 3.89). General-ability NLL rises **more** with JEPA (epoch 20: 1.10 against 1.03). The JEPA loss mostly **collapses**: mismatched-pair cosine is nearly the matched one (0.979 against 0.984 at λ 4). Blind at epoch 10, plain LoRA against JEPA λ 4: 25 to 36, 39 ties (p = 0.2).
- **300 pairs:** pending.
- **Conclusion so far:** on this task and scale, JEPA does not beat a tuned LoRA. Its measurable effect is a regulariser that delays overfitting. Decision (user, 2026-11-05): record JEPA and move Project 4 to "LoRA the hard way", with two personalities (Titan and Salim–Javed).
