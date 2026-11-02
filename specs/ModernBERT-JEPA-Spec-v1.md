# ModernBERT-JEPA Decider: technical specification v1

Status: draft for build (2026-11-02). It refines the first-pass note in `ModernBERT-JEPA-Developer-Specification.md` (kept unchanged next to this file). Nothing here has been run yet.

---

## 0. Summary

**What we build:** a small encoder decision model (ModernBERT-base, 149M) that answers typed decision questions (`noul`, `choice`, `score`) with a probability for every option, in one encoder pass per question, with no generated text.

**How it learns:** with a **joint-embedding predictive** objective. A context encoder reads the question and the state. A predictor maps that reading to a latent "answer vector". A slowly moving copy of the encoder (EMA) embeds each option. The prediction must land near the embedding of the right option, in latent space, not in token space. Option probabilities are a softmax over the similarities between the prediction and each option's embedding. That works for any number of options, any names and any question, which a general model needs.

**Two targets, one codebase, two heads:**

| | Head A: Typed Decisions | Head B: JevBench |
|---|---|---|
| Benchmark | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), zero-shot table | [JevBench](https://benchmarkheaven.com/jev-models), open-weights board |
| States | short JSON records (median about 720 chars, max about 2,000) | mostly prose; the hard tier is long documents (1,600 to 15,300 chars) with traps |
| Gold | soft (mean of 3 teacher samples); KL, Brier, ECE | hard (one expected label); accuracy, Brier, ECE, ordinal MAE, paraphrase consistency |
| Context length | 1,024 tokens | 8,192 tokens (ModernBERT's native limit); chunked beyond that |
| Pooling | mean pool over the question and the state | question-conditioned attention pooling over state tokens (finds the deciding clause) |
| Loss emphasis | match soft distributions (KL) | sharp but calibrated (label-smoothed CE plus calibration) |

Stages 1 and 2 of training are shared. Stage 3 and the head are per benchmark.

**Our yardsticks:**
- **Typed Decisions:** our 4B decoder `skamalj/signpost-d-4b` scores 0.712 / KL 0.162. The best small encoder, `tasksource-decider-nano` (150M), scores 0.7515, but in the fitted table.
- **JevBench:** Qwen3.5-4B fine-tunes score 55 to 75 Capability.

A 149M encoder that reaches **≥ 0.68 on Typed Decisions in general mode** at about 20× lower cost and latency would be a real result. Beating 0.712 would be a strong one.

---

## 1. Goals, success criteria, non-goals

**Goals**
1. A **general** model: any question schema at request time; never trained on either benchmark's evaluation items.
2. Served through a `/v1/systemone` endpoint in the TypeSafe wire format, so both benchmarks' harnesses can call it unchanged.
3. Latency below 30 ms per case (5 questions) on one GPU, and usable on CPU.
4. Training within about 2 GPU-hours per seed on a Colab G4 (RTX PRO 6000, 96 GB).

**Success criteria, each judged before any test run**
- **Typed Decisions, Head A:** on the benchmark's `train` split (never trained on), accuracy ≥ 0.68 and KL ≤ 0.20. A stretch goal is ≥ 0.712, our 4B.
- **JevBench, Head B:** on the public items scored with JevBench's own harness (`original` 72, `easy` 48, `hard` 111), accuracy ≥ our 4B's on the same items. Valid answers must be 100%.
- **JEPA must earn its place:** the full model beats the same architecture trained with the latent loss switched off (λ_jepa = 0) by ≥ 1.5 points, over three seeds.

**Non-goals**
- A fitted or specialist model (trained on Typed Decisions' `train` split). That's a separate track.
- Generating text, or reasoning at inference.

---

## 2. Rules carried over (non-negotiable)

- **Typed Decisions zero-shot rule:** no `train` or `test` rows, labels, workflows or question schemas in training. The `train` split may be used only to **compare candidates**, never to train, fit or calibrate (no temperatures fitted on it).
- **One `test` run per released model,** at the end, after the choice is made on the `train` split. Every run is disclosed.
- **JevBench public items are never trained on.** They're our only local JevBench check. JevBench requires disclosure if anyone trains on them.
- **Contamination filter on every external source:** drop any row from or derived from LocalLLaMA/typed-decisions, JevBench, or the Decision Index evaluation sets. Each filter's counts go into the run log.
- **Public wording** (cards and posts): name no generator or judge models; no history before the release line; never claim the data was uninfluenced by test results.
- **No local runs** (WSL crashes): notebooks run on Colab or Kaggle. Locally we do static checks only (compile, pyflakes).
- **Commits** as `kamal <skamalj@gmail.com>`, with no AI co-author lines.

---

## 3. What we learned before (and how this design answers it)

| Earlier finding | Consequence here |
|---|---|
| Our old ModernBERT-large decision encoders reached only 0.45–0.52 on Typed Decisions, below the prior (0.470), with dev at about 0.70. They answered from signal words and didn't read the deciding numbers or clauses. | The model has to be made to read: question-conditioned pooling (Head B), much more and more varied data (Stage 1), and distillation from a model that does read (Stage 2b). |
| They were trained on about 2,000 cases of our own data. | Data scale is the main lever: `tasksource-decider-nano` (150M) reached 0.7515 with 480k requests. |
| Collapse to uniform output near the LR peak; ModernBERT overflows in fp16 (it was pre-trained in bf16). | **bf16** on the G4 (fp32 master weights); clip gradients at 1.0; log the gradient norm; an automatic collapse check at step 400; layer-wise LR decay. |
| Our dev split didn't predict the benchmark. | Selection uses the Typed Decisions `train` split (as for the 4B), plus JevBench's public items for Head B. |
| The 4B soup (3 seeds, α 0.75) added calibration. | The same 3-seed soup step, cheap at 149M. |

---

## 4. Task interface

**Input:** one System One request: `state` (a string or JSON), `questions` (`{qid: {type, instructions, criteria}}`).

**The options of a question**, each turned into a short **option text**:

| Type | Options | Option text |
|---|---|---|
| `noul` | `true`, `false` | `criteria.true` / `criteria.false` if given, else "The statement is true." / "The statement is false." |
| `choice` | `criteria` keys | `"{name}: {description}"`; underscores become spaces in the name |
| `score` | the levels, in order | `"Level {i} of {L-1}: {level text}"` |

**Context text** (one per question): `"Question: {instructions}\n\nState:\n{state as compact JSON or text}"`. The question comes first, so truncation never cuts it.

**Output per question:** a probability for every option, summing to 1.

**Wire format** (both harnesses read it; it's what vLLM's `/v1/systemone` returns):
- `noul`: `{"type": "noul", "noul": p_true, "probabilities": {"yes": p_true, "no": p_false}, "confidence": ...}`
- `choice`: `{"type": "choice", "choice": top_name, "probabilities": {name: p, ...}, "confidence": ...}`
- `score`: `{"type": "score", "score": expected_level, "legend": {"0": text, ...}, "probabilities": {"0": p, ...}, "confidence": ...}`

The response is `{"id", "object": "structured_decision", "model", "answers": {qid: answer}, "usage": {"input_tokens", "output_tokens": 0}}`.

---

## 5. Architecture

### 5.1 Components

```
 context text ─► E_θ  (ModernBERT-base, trained)      ─► token states H_c ─► pool ─► z_c (768)
                                                                                  │
                                                                     P_φ (predictor MLP)
                                                                                  ▼
                                                                         ẑ (768, L2-normalised)
 option text k ─► E_ξ  (EMA copy of E_θ, no gradient) ─► mean pool ─► proj_ξ ─► y_k (768, L2-normalised)

 s_k = cos(ẑ, y_k) / τ_type          p = softmax_k(s_k + b_type,k)
```

- **E_θ: the context encoder.** `answerdotai/ModernBERT-base`, all 22 layers trainable with layer-wise LR decay of 0.85 from the top. The embeddings stay frozen. (The first-pass note froze 18 layers; at 149M a full fine-tune is cheap, and the old line showed that frozen readers don't learn to read.)
- **E_ξ: the target encoder.** An exponential moving average of E_θ (momentum 0.996 rising to 0.9999 on a cosine schedule). It has no gradients, and it embeds option texts only. This is the JEPA's target network: it gives a stable latent space to predict into, and it prevents the trivial solution where the encoder learns to map everything to one point.
- **P_φ: the predictor.** `Linear(768→1536) → GELU → LayerNorm → Linear(1536→768)`, plus a residual from z_c. It predicts where the right answer sits in the target space, given the reading.
- **proj_ξ:** a `Linear(768→768)` on the target side, as an EMA of a trained `proj_θ` (the BYOL/I-JEPA pattern).
- **τ_type:** a learned temperature per type (`noul`, `choice`, `score`), initialised at 0.07.
- **b_type,k:** an optional learned bias per position, used only for `noul` (yes/no prior) and `score` (an ordinal prior over position). It's zero for `choice`, so option names carry no prior.

Size: about 149M (encoder) + 3.5M (predictor and projection). The EMA copy is only needed for training and export; at inference it supplies the option embeddings.

### 5.2 Why this is a JEPA, and not just a classifier

The objective is a **prediction in latent space**: from the context, predict the target encoder's embedding of the answer. That's the I-JEPA / LLM-JEPA pattern (context encoder, predictor, EMA target encoder, a loss in embedding space), applied with question → answer as the masked "target view". The classifier head of the first-pass note is replaced by **similarity to the option embeddings**. That's what makes the model work for options it has never seen.

**Collapse guards:**
1. The EMA target with stop-gradient.
2. A **variance-covariance regulariser** on ẑ over the batch (VICReg-style: keep each dimension's std ≥ 1 and decorrelate dimensions), with a small weight.
3. The decision loss itself, which needs options to be separable.

All three are logged (the std of ẑ, the average pairwise similarity of the y_k).

### 5.3 Head A, Typed Decisions ("pooled")

- **Context:** up to 1,024 tokens. Typed Decisions states are at most about 2,000 characters, so nothing is cut.
- **Pooling:** mean over the context tokens, giving z_c.
- **Case batching:** a case's 5 questions are 5 context sequences of about 300 tokens, run as one batch. That's fast on an encoder, so the state isn't shared through tree attention in v1. Tree attention, as in `tasksource-decider-nano`, is a later speed option.

### 5.4 Head B, JevBench ("evidence")

- **Context:** up to 8,192 tokens. Longer states are split into overlapping 4,096-token windows, and the window whose pooled latent has the highest predicted margin wins.
- **Pooling, question-conditioned:** the question tokens' mean is a query that attends over the state tokens (one attention layer, 8 heads). The pooled state is concatenated with the question mean, then projected to z_c. This lets the model weight the deciding clause rather than the average of a long document: the old encoder's failure mode.
- **Calibration:** the same per-type temperatures, plus label smoothing. JevBench's gold is a single label, and ECE is half of its Calibration score.

### 5.5 Baseline control C0 (to prove JEPA's value)

The same encoder, scoring each option as a **cross-encoder** `(context, option text) → logit`, trained with cross-entropy only: the tasksource-nano design. Every milestone compares JEPA (C2) against C0 and C1 (JEPA architecture with λ_jepa = 0).

---

## 6. Losses

For a question with options k = 1..K, gold distribution g (soft, or one-hot with smoothing), and model distribution p:

- **Decision:** L_dec = KL(g ‖ p) = Σ_k g_k log(g_k / p_k). For hard gold, g = (1 − ε)·onehot + ε/K with ε = 0.05 (Head B).
- **JEPA (latent prediction):** ȳ = normalise(Σ_k g_k · sg(y_k)), the gold-weighted target embedding with stop-gradient. Then **L_jepa = 1 − cos(ẑ, ȳ)**. A Smooth L1 on the normalised vectors is an ablation. This is the first-pass note's latent loss, with its target now defined.
- **Ordinal (`score` only):** L_ord = |Σ_k k·p_k − Σ_k k·g_k| / (K − 1), the expected-level error that helped the 4B.
- **Regulariser:** L_vc = VICReg variance + covariance terms on the batch of ẑ.

**Total:** L = L_dec + λ_jepa·L_jepa + λ_ord·L_ord + λ_vc·L_vc, starting at λ_jepa = 0.5, λ_ord = 0.5, λ_vc = 0.04.

---

## 7. Data

### 7.1 Stage 1, broad alignment (shared by both heads)

- **Source:** `tasksource/tasksource-jev-typed-decisions` (1M–10M rows; `state`, `question`, `options`, `target` (a distribution), `kind`).
- **Before use:**
  1. **Licence:** the card says `other`. List the sources inside it and keep only those whose licences allow training a released Apache-2.0 model. Record the kept list.
  2. **Contamination:** drop every row whose source is LocalLLaMA/typed-decisions, JevBench, or any Decision Index evaluation set. Also drop near-duplicates of their states (normalised 8-gram overlap ≥ 0.5 against the TD `train` and `test` states and JevBench's public items). Log the dropped counts per source.
  3. **Type mapping:** map its `kind` values to `noul`, `choice` or `score`, and drop extraction kinds (they're span tasks, not decisions).
- **Size:** a balanced sample of about 300k questions, capped per source.

### 7.2 Stage 2, our generated data (shared)

- **Source:** `skamalj/signpost-d-decisions` (private): `train` 2,575 records / 12,274 questions with soft gold from three simulated judges; `dev` 200 records from 8 domains not in `train`.
- **Use:** all of `train`, for 3 epochs; `dev` for the collapse check and epoch choice.

### 7.3 Stage 2b, distillation from our 4B (shared, optional, likely the biggest gain)

- **Teacher:** `skamalj/signpost-d-4b` served with vLLM `/v1/systemone`, about 60 ms per case.
- **Pool:** unlabelled states with their questions: the Stage-1 states (after the filter) and newly generated records. No benchmark states.
- **Target:** the teacher's distribution, mixed with the original gold where there is one (0.5/0.5), and with gold only where they agree on the top option.
- **Size:** 200k–500k questions. At about 12 ms per question on a G4 that's 1–2 GPU-hours, with no API cost.

### 7.4 Stage 3, per head

- **Head A:** Stage 2 data again at a low LR, plus a 20% replay of Stage 1. The aim is calibration toward soft, teacher-like distributions.
- **Head B:** **long-document records**, a new generation round, briefed from the *patterns* of JevBench's hard tier, never its items or domains:
  - long policy, contract or roster text;
  - the deciding clause far from a misleading one;
  - definitions that overturn the obvious answer;
  - date windows, sublimits, counts.

  About 2,000 records, using the same generator and judge pipeline as `signpost-d-decisions`, within a **USD 50** budget (user, 2026-11-02).

### 7.5 Never used for training

The Typed Decisions `train` and `test` splits; JevBench's public items; any Decision Index evaluation set.

---

## 8. Training recipe

| Setting | Stage 1 | Stage 2 / 2b | Stage 3 |
|---|---|---|---|
| Precision | bf16 autocast, fp32 master weights | same | same |
| Optimizer | AdamW, weight decay 0.01 | same | same |
| LR, encoder top layer (LLRD 0.85 down) | 5e-5 | 3e-5 | 1e-5 |
| LR, predictor / projection / temperatures | 1e-3 | 5e-4 | 2e-4 |
| Schedule | warm-up 5%, cosine decay | warm-up 5%, cosine | warm-up 10%, cosine |
| Batch | 64 questions (with all their options) | 32 | 32 |
| Epochs | 1 pass over about 300k | 3 (2b: 1 pass) | 2 |
| Max context tokens | 1,024 | 1,024 | A: 1,024 / B: 8,192 |
| EMA momentum | 0.996 → 0.9999 | continue | continue |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Seeds | 42 | 42, 1042, 2042 | 42, 1042, 2042 |

- **Run time:** an estimate, to be measured in M0 on the G4: Stage 1 about 40 min, Stage 2 about 5 min, Stage 2b about 30 min, Stage 3A about 5 min, Stage 3B about 20 min.
- **Checkpoints:** a resume point every 500 steps, and an adapter or full checkpoint per stage. A collapse check at step 400 of every stage: the dev KL must beat the uniform answer's KL by ≥ 0.02. If it doesn't, restart with the next seed.
- **Soup:** average the three seeds' full weights (they all start from the same Stage-1 checkpoint, so they stay in one basin). Interpolate α ∈ {1.0, 0.75} toward the Stage-1 weights; choose on the Typed Decisions `train` split (Head A) or the JevBench public items (Head B).

---

## 9. Evaluation and selection

**Our scorers**
- **Typed Decisions:** the benchmark's own `score_one`, copied unchanged from `bench/score_with_jobs.py` (as in 02), run against our server on the `train` split: overall, per type and per workflow/question.
- **JevBench:** their harness (`python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' --cost-basis no_billable_account_public_endpoint --reserve-usd 0`, then `summarize`) on `original`, `easy` and `hard`.
- **Our own dev:** collapse and epoch checks only.
- **JEPA diagnostics:** cos(ẑ, ȳ) on dev; the std of ẑ; the mean y_k–y_j similarity; and a "rank of the gold option among all the dev set's option embeddings" probe.

**Milestones and gates**

| | What | Gate to continue |
|---|---|---|
| **M0** | The pipeline end to end on 2k Stage-1 questions: server, both scorers, timings | everything runs; valid answers 100% |
| **M1** | **C0** (cross-encoder) and **C1** (JEPA architecture, λ_jepa = 0), Stage 1 + 2, seed 42 | TD `train` split ≥ prior (0.470) by a clear margin |
| **M2** | **C2**, the full JEPA, Stage 1 + 2, seed 42 | C2 ≥ C1 + 1.5 points on the TD `train` split; if not, JEPA isn't helping and we stop or rethink |
| **M3** | + Stage 2b distillation | TD `train` split up ≥ 2 points over M2 |
| **M4** | Head A, Stage 3, 3 seeds + soup | TD `train` split ≥ 0.68 / KL ≤ 0.20 |
| **M5** | Head B, long-document data + Stage 3B, 3 seeds + soup | JevBench public ≥ our 4B on the same items |
| **M6** | Release: one TD `test` run (HF Jobs), the JevBench issue | – |

---

## 10. Serving

- **`serve.py`:** readable Python (FastAPI and uvicorn), with no compiled or binary code. JevBench's maintainer refuses code they can't read (their issue #228).
  - `POST /v1/systemone`: parses the request, renders the contexts and option texts (§4), encodes the contexts with E_θ and the options with E_ξ, scores them (§5) and returns the wire format.
  - `GET /health` is also provided.
- **Option-embedding cache:** an LRU keyed by option text, since option sets repeat.
- **Batching:** one request = all its questions in one forward pass; serial requests, as JevBench measures.
- **Precision:** bf16 on GPU, fp32 on CPU. ONNX fp16 export is optional and later, only if latency needs it. The JevBench cap is 1.23 s, so plain PyTorch is enough.
- **Packaging:** the model repo holds the weights (E_θ, E_ξ, predictor, projection, temperatures and biases), `config.json`, `serve.py`, `requirements.txt` with pinned versions, and a one-command start: `pip install -r requirements.txt && python serve.py --model . --port 8000`.

**Typed Decisions official scoring:** the benchmark's `score_with_jobs.py` starts **vLLM**, which can't serve this architecture. So:
1. Run our server and a client that applies the benchmark's own scoring code unchanged (`score_one`, the ECE bins), on HF Jobs or Colab.
2. Write `results.json` and the per-decision predictions.
3. Open the `.eval_results` PR with the benchmark's `submit_results.py`.

The maintainers rescore published predictions (they did for #61), so the predictions file is required.

---

## 11. Release

- **Repos:** `skamalj/signpost-jepa-base-td` (Head A) and `skamalj/signpost-jepa-base-jev` (Head B), or one repo with two configs. To decide at M6.
- **Cards** follow the signpost-d-4b card's structure and wording rules. Training disclosure:
  - the Stage-1 source and its filter counts;
  - our generated data;
  - distillation from our own 4B;
  - the number of `test` runs.
- **Typed Decisions:** a discussion in the same format as #61.
- **JevBench:** a GitHub issue "[bench request]: …" with the pinned revision, the server command, the cost basis (149M class), "one encoder pass per question, no ensembles", and the training disclosure.

---

## 12. Files and notebooks (branch `decider-jepa`)

| File | Purpose |
|---|---|
| `jepa/model.py` | E_θ, the EMA target E_ξ, predictor, projection, both pooling heads, scoring |
| `jepa/data.py` | System One → contexts and option texts; the Stage-1 loader with licence and contamination filters |
| `jepa/losses.py` | L_dec, L_jepa, L_ord, L_vc |
| `jepa/serve.py` | `/v1/systemone` server (also shipped in the model repo) |
| `01_jepa_data.ipynb` | Stage-1 filtering report; Stage-2b distillation pool (serving the 4B with vLLM) |
| `02_jepa_train.ipynb` | stages 1–3 for C0, C1 and C2, seeds, collapse checks, soup |
| `03_jepa_evaluate.ipynb` | server + TD `train`-split scorer + JevBench harness; diagnostics; the one `test` run at release |

Notebooks follow the Signpost-D style: small cells, each preceded by a markdown explanation.

---

## 13. Risks and open questions

**Risks**
1. **A small encoder may not reach the 4B's reading ability.** The old line failed on that. Distillation (2b) and Stage-1 scale are the mitigations; M1–M3 tell us early.
2. **JEPA may not beat plain training (C2 ≈ C1).** That's a legitimate finding: we'd report it and keep the better design.
3. **Stage-1 licences:** if too much of the tasksource data is unusable for a released model, Stage 1 shrinks and distillation carries more of the load.
4. **Long documents (Head B):** 8,192 tokens covers the public hard tier (≤ about 4k tokens). Their private items may be longer, and windowing is untested.

**Decisions (user, 2026-11-02)**
1. **ModernBERT-large:** not yet. Base only; large can come later as an ablation.
2. **Distillation from our 4B (Stage 2b):** yes, as part of the general model.
3. **Head-B long-document generation budget:** USD 50 in total, all stages included. Plan about 2,000 records within it, and stop if a pilot shows it won't fit.
4. **Branch:** `decider-jepa`, a new standalone branch from `master`, like the other parts. Suggested; to be created when the build starts.

---

## Appendix: changes from the first-pass note

| First-pass note | v1 | Why |
|---|---|---|
| Target latents undefined | the EMA target encoder's embedding of the gold-weighted option (§5.2, §6) | without a target there is no JEPA |
| `num_routes = 64` fixed head | similarity to per-option embeddings, any K | both benchmarks change options per question |
| Argmax output | a full distribution in the System One wire format | both benchmarks score probabilities (KL, Brier, ECE) |
| CE in the formula, missing in the code | KL / label-smoothed CE implemented, plus ordinal and VICReg terms | completeness; calibration |
| 18 of 22 layers frozen | full fine-tune with layer-wise LR decay | at 149M it's cheap; frozen readers didn't learn to read |
| L = 128 tokens | 1,024 (Head A) / 8,192 (Head B) | TD states reach 2k chars; JevBench hard items about 4k tokens |
| fp16 | bf16 on the G4 | ModernBERT overflows in fp16 (an old collapse cause) |
| ONNX for a 15 ms target | plain PyTorch first; ONNX optional | the JevBench cap is 1.23 s; readable code matters more |
| JevBench only | two heads: Typed Decisions + JevBench | the user's requirement |
| Training data unspecified | Stage 1 tasksource (filtered), Stage 2 ours, 2b distillation, 3 per head | data scale is what made the 150M tasksource model work |
