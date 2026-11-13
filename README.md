# LLMs the Hard Way

Jupyter notebooks that open up real language models and rebuild their internals by hand, then put them to work. It starts with **decoders** (GPT-2 and SmolLM2: attention, RoPE, the KV cache, LoRA) and continues with an **encoder** (ModernBERT). From there it builds a **multi-agent router** and ends with a published model, **[Signpost](https://huggingface.co/skamalj/signpost)**: it routes each chat message to the right agent and answers grounded yes / no, choice and score questions about a text.

Every step computes things manually first and then checks the result against the Hugging Face implementation.

## Which branch holds what

Each part of the series lives on its own branch. **This README is the same on every branch**, so you can start anywhere.

| Part | Branch | Contents | Runs on |
|---|---|---|---|
| 1. Decoders by hand | [`master`](https://github.com/skamalj/llms-the-hard-way/tree/master) | `kv_cache_experiment.ipynb`, `smollm2_forward_pass_kv_cache.ipynb`, `rope.ipynb`, `smoll_json_lora.ipynb` | Laptop CPU |
| 2. An encoder by hand | [`modernbert-by-hand`](https://github.com/skamalj/llms-the-hard-way/tree/modernbert-by-hand) | `01_modernbert_by_hand.ipynb` | CPU (or Kaggle) |
| 3. A frozen-encoder router | [`frozen-router`](https://github.com/skamalj/llms-the-hard-way/tree/frozen-router) | `02_frozen_router.ipynb`, its data, the experiment ledger | Kaggle GPU |
| 4. Fine-tuning: Signpost | [`signpost`](https://github.com/skamalj/llms-the-hard-way/tree/signpost) | `03_lora_signpost.ipynb`, `04_decider_benchmark.ipynb`, `05_one_pass_signpost.ipynb` (4b), `signpost/` (loaders + model card), their data, the experiment ledger | Kaggle or Colab GPU |
| 5. Signpost in agent frameworks | [`signpost-apps`](https://github.com/skamalj/llms-the-hard-way/tree/signpost-apps) | `apps/langgraph/`: five agents routed by Signpost, with a DynamoDB checkpointer and a message reducer | Local (CPU), AWS, Anthropic API |

The suggested reading order is 1 → 5.

### How the series flows

Each part builds what the next one needs:

1. **Decoders by hand** (`master`): attention, RoPE and the KV cache rebuilt by hand on GPT-2 and SmolLM2, then a first LoRA (SmolLM2 answering in JSON).
2. **An encoder by hand** (`modernbert-by-hand`): ModernBERT's forward pass rebuilt from its checkpoint, and a classification head on top. A new class needs new labels and a retraining run, so…
3. **A frozen-encoder router** (`frozen-router`) changes the head: the encoder stays frozen and small heads score each candidate agent against the conversation, so a new agent needs only its description. The frozen encoder turns out to be the limit, so…
4. **Fine-tuning: Signpost** (`signpost`) fine-tunes ModernBERT-large with LoRA into one model for routing and grounded questions (Signpost), benchmarks it against the 2B Strands decider, and in 4b scores every option in one pass.
5. **Signpost in agent frameworks** (`signpost-apps`) puts Signpost to work as the router of a real multi-agent app.

The notebooks are numbered across the series (`01` in Part 2 to `05` in Part 4); Part 1's notebooks are unnumbered.

## Part 1: Decoders by hand (`master`)

Small, CPU-friendly notebooks. The models are tiny (a 2-dimensional GPT-2 and the 135M-parameter SmolLM family), so every tensor can be printed and inspected.

| Notebook | Model | What it covers |
|---|---|---|
| [`kv_cache_experiment.ipynb`](https://github.com/skamalj/llms-the-hard-way/blob/master/kv_cache_experiment.ipynb) | `sshleifer/tiny-gpt2` | A complete GPT-2 forward pass by hand on a model with 2-dim embeddings: token + position embeddings, LayerNorm, the packed `c_attn` Q/K/V projection, multi-head split, causal mask, softmax, residuals, the GELU MLP, both blocks, `ln_f` and the LM head. Ends with an end-to-end check that the manual result matches the model. |
| [`smollm2_forward_pass_kv_cache.ipynb`](https://github.com/skamalj/llms-the-hard-way/blob/master/smollm2_forward_pass_kv_cache.ipynb) | `HuggingFaceTB/SmolLM2-135M` | The same walkthrough for a modern Llama-style model: RMSNorm, grouped-query attention (9 query / 3 KV heads), RoPE, `repeat_kv` and the SwiGLU MLP. Layer 0 is traced by hand and layers 1–28 run through `layer.forward()`. The last layer runs as a **KV-cache decode step**: cached K/V for earlier tokens, and fresh Q/K/V only for the new token. Final logits match Hugging Face to within about 3e-5, and the notebook works for any prompt length. |
| [`rope.ipynb`](https://github.com/skamalj/llms-the-hard-way/blob/master/rope.ipynb) | `HuggingFaceTB/SmolLM2-135M` | Rotary position embeddings from scratch: the 32 rotation pairs of a 64-dim head, the `inv_freq` frequencies, per-position angles, rotating one pair and then a whole head by hand, length preservation, an exact match with `apply_rotary_pos_emb`, and why RoPE makes attention depend on relative position. |
| [`smoll_json_lora.ipynb`](https://github.com/skamalj/llms-the-hard-way/blob/master/smoll_json_lora.ipynb) | `HuggingFaceTB/SmolLM2-135M` + LoRA | Fine-tunes a LoRA adapter (r = 8, attention projections only) so the base model answers a question with a single `{"answer": "..."}` JSON object and then stops. Covers data splitting, prompt masking, EOS handling, training with 🤗 `Trainer`, base-vs-LoRA evaluation and optional debugging cells. |

**Setup.** Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/skamalj/llms-the-hard-way.git
cd llms-the-hard-way          # master

uv sync
uv pip install peft datasets accelerate matplotlib   # used by the RoPE and LoRA notebooks

uv run jupyter lab
```

The models download from the Hugging Face Hub on first use. No GPU is needed. Setting `HF_TOKEN` is optional and only raises the download rate limit.

**The LoRA notebook.** `smoll_json_lora.ipynb` is controlled by a single **run configuration** cell at the top:

| Setting | Purpose |
|---|---|
| `DATA_FILE` | Training data, as a JSON list of `{"prompt": "...", "answer": "..."}` records |
| `TRAIN_FRAC` / `VAL_FRAC` / `TEST_FRAC`, `SEED`, `DEDUPLICATE` | How the data is split |
| `RUN_TRAINING` | `False` skips training and evaluates the adapter already at `ADAPTER_PATH` |
| `ADAPTER_PATH` | Where the adapter is saved and loaded from |
| `RUN_DEBUG` | Turns the diagnostic cells at the end on or off |

The included dataset, `data/lora_json_answer/smollm2_answer_250_fresh.json`, has 250 short technical Q&A pairs. Training for 10 epochs on its 200-record training split takes about 7 minutes on a CPU. Not committed: the trained adapters (`smollm2-*-lora*/`) and the train/validation/test split files, which the notebook regenerates from `data/`.

## Part 2: An encoder by hand (`modernbert-by-hand`)

**`01_modernbert_by_hand.ipynb`** builds the forward pass of [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) (28 layers, H = 1,024) from its real checkpoint, loaded as a plain dict of tensors. Attention itself (Q / K / V, heads, RoPE) is built step by step in Part 1, so this notebook recaps it briefly and focuses on what an encoder changes:
- tokenization and embeddings;
- **bidirectional** attention (no causal mask), compared with SmolLM2's causal attention;
- local (sliding-window) vs global layers, each with its own RoPE base (using ModernBERT's own RoPE);
- LayerNorm without bias, residuals and the GeGLU MLP;
- the full 28-layer stack.

It matches the reference `last_hidden_state` to about 3e-5. Then comes the key idea: a small **classification head** on the pooled encoder vector. The notebook trains one on 6 customer-message intents (8 examples each, encoder frozen) and tests it on unseen messages. Then it adds a 7th intent: the new class needs its own labelled examples and a retraining run. That's why Part 3 changes the head.

**Run:** top to bottom, on a CPU or on Kaggle / Colab (the first cell installs the pinned `transformers`).

## Part 3: A frozen-encoder router (`frozen-router`)

**`02_frozen_router.ipynb`** keeps the encoder **frozen** and trains small heads on top. For each candidate agent, one pass reads (conversation + current agent + agent description), and a head scores the fit:
- a cross-encoder router;
- a yes / no "has the topic changed?" detector, used as a one-sided "confident stay" gate;
- a choice head for grounded questions.

It's tested on unseen banking agents, on realistic multi-turn chats (E10), on first messages (E11) and on grounded questions (E12); the topic-change detector is also tested on E10's labelled turns. The comparison model is the 2B [Strands decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19).

The notebook ends with **what we tried and dropped**, with the numbers: hard cascade, soft fusion, Q-K-V attention head, windowing, layer mix, confident switch, and more. The best frozen system routed conversations at unseen companies better than the decider (74 vs 62 of 83 turns), but grounded questions plateaued at 114 / 170 against its 160. **The frozen encoder was the limit.**

**Run:** Kaggle with a GPU; the data is cloned from this branch. **Ledger:** `experiments/modernbert_router_ledger.md` records every run, with what changed, the results and the decision. Runs 1–25 are on both `frozen-router` and `signpost`; each branch's copy then adds its own later work (the notebook reruns, and on `signpost` the score-data attempt).

## Part 4: Fine-tuning, and Signpost (`signpost`)

**`03_lora_signpost.ipynb`** adds **LoRA** adapters (r 16, 7.2 M parameters, through the `peft` library; a by-hand LoRA notebook is planned) to every attention and MLP layer of ModernBERT-large, plus a 0.26 M scoring head. It trains **one model for routing and grounded questions**, with one forward pass per option and an order-aware loss for score questions, tests it on every set (including two independent ones), then saves it, merges LoRA into the weights and checks that the published loader reproduces the results exactly.

It runs Signpost's final recipe; the earlier experiments (Nemotron-1B, separate specialists, plain vs ordinal loss) are reproducible from their own commits. It ends with **what we tried and dropped**: more score data, Nemotron as the final encoder, separate specialists, a training collapse with its guard, and hand-written score data on 279 scales (no gain).

**`04_decider_benchmark.ipynb`** runs the 2B Strands decider on exactly the same test items: E10, E11, banking, E12, both independent sets, and latency.

**`05_one_pass_signpost.ipynb`** (Part 4b) puts **every option into one input**, so the encoder runs once per question instead of once per option, and one shared head scores each option at its own tokens. Trained on Signpost's data plus 2,500 hand-written many-option questions and routing with up to 15 agents, it is level with Signpost on routing (independent 134 vs 133 / 172, many-agent 102 vs 103 / 120), slightly behind on E12 (152–158 vs 161), and its latency stays flat at about 60 ms for 2–50 options, where Signpost's grows with each option (about 190 ms with 8–11 agents). It ends with runs 26–29 of the one-pass experiment.

**`signpost/`** holds the standalone loaders (`signpost.py`; `signpost_one_pass.py` for the one-pass model), the model card and its tested examples. The exported model is published at **[huggingface.co/skamalj/signpost](https://huggingface.co/skamalj/signpost)** (Apache-2.0). Tag `signpost-v1.0` marks the exact state that trained and exported it.

| Same test items for both | Signpost | Strands decider 2B |
|---|---|---|
| Routing in conversations, unseen companies | **78 / 83** | 62 / 83 |
| First message | **97 / 100** | 96 / 100 |
| Grounded questions (E12) | **161 / 170** | 160 / 170 |
| Independent questions (separate author) | 221 / 300 | **253 / 300** |
| Independent routing in conversations | **133 / 172** | 113 / 172 |
| Independent set 2: questions · routing with 8–11 agents | **188 / 200** · **62 / 70** | 185 / 200 · 55 / 70 |
| Latency per decision (T4, p50) | **~55–75 ms** | ~200 ms per question, 290–650 ms per routing decision |

**Run:** Kaggle or Colab with a GPU (all three notebooks; the latest saved outputs are from Colab, RTX PRO 6000); the data is cloned from this branch.

## Part 5: Signpost in agent frameworks (`signpost-apps`)

**`apps/langgraph/`** is a customer-support app for a fictional retailer: five agents (Claude) with Signpost as the router node. Chat memory is stored in DynamoDB through `langgraph-dynamodb-checkpoint`, whose reducer prunes old messages. The tests cover:
- first messages;
- live conversations;
- resuming a thread from DynamoDB;
- routing from a pruned history.

The result was 67 / 71 turns routed correctly. More frameworks (CrewAI, Strands Agents) can follow in `apps/`.

**Run:** see `apps/langgraph/README.md` (uv, an AWS profile for DynamoDB, an Anthropic API key). Signpost downloads from the Hugging Face Hub.

## Tags

| Tag | Marks |
|---|---|
| `forward-pass-walkthrough-v1` | SmolLM2 forward pass by hand, ending in a KV-cache decode step |
| `rope-walkthrough-v1` | RoPE walkthrough with a verified exact match against Hugging Face |
| `lora-json-answer-round3` | LoRA JSON-answer adapter, round 3 settings and results |
| `signpost-v1.0` | The state that trained and exported the published Signpost model (run 25) |
| `signpost-one-pass-v1.0` | The run of `05_one_pass_signpost.ipynb` that trained and exported the published [Signpost one-pass](https://huggingface.co/skamalj/signpost-one-pass) model |
| `archive/exp/*` | Experiment branches whose commits are in no part branch (the original decider benchmark, an encoder-combination run, and the one-pass runs 26–29 with their outputs) |

The router work began as one `exp/*` branch per experiment (runs 5–29). Those branches are consolidated into the ones above and deleted; every commit is still reachable from a part branch or an `archive/exp/*` tag. Each ledger entry names its run and original branch.

## License

The Signpost model is Apache-2.0, fine-tuned from [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) (Apache-2.0).
