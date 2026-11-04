"""Build 10_td_jepa_2b.ipynb: Qwen3.5-2B trained with Signpost-D's recipe, without (D1) and with (D2) a JEPA
pointer, scored on Typed Decisions' test split; T scores Signpost-D-4B there with the same harness."""
import json
import uuid

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n")))


def code(text):
    cells.append(("code", text.strip("\n")))


md(r'''
# Typed Decisions with JEPA: Qwen3.5-2B

Signpost-D-4B scores 0.712 on Typed Decisions' test split (zero-shot: trained on our own records only). This notebook asks whether a smaller model, Qwen3.5-2B, gets further when JEPA is added to the same recipe.

**The recipe (Signpost-D's, unchanged).** One question per sequence in vLLM's exact prompt (the state, the question with labelled options, the reply line); the answer is the reply's first token, softmaxed over the option labels. LoRA rank 16 on the attention, MLP and linear-attention projections; soft cross-entropy against our judges' distribution (plus the expected-level distance for score questions); 2 epochs, two records per step, learning rate 1e-4 with a 15% warm-up and linear decay. Trained on our 2,575 records only: no Typed Decisions data.

**The JEPA part (D2 only).**
- **Target:** an EMA copy of the model (its LoRA weights follow the trained ones) reads the same prompt; its hidden state at the **end of each option's line** is that option's target (in a decoder, the line end is where the option has been read whole). The targets of one question are centred, so only what distinguishes the options remains.
- **Predictor:** a narrow transformer (`PRED_LAYERS` layers, width `PRED_WIDTH`) over the model's whole output, read at the answer position. **D2:** ẑ = answer vector + predictor output (the last projection starts at zero). **D3:** ẑ = the predictor's output alone (normal start, learning rate `LR_PREDICTOR_D3`): the answer vector is large and points elsewhere, so in D2 it dominated the cosine with the small centred targets and the JEPA loss could hardly move.
- **Losses:** ẑ is pulled onto the gold option's centred target (1 − cos, weighted by the gold distribution); a pointer readout, softmax of cos(ẑ, targets) / τ over the question's options, is trained with KL to the gold.
- **The head is unchanged:** the label tokens' logits at the answer position, the readout vLLM serves.

**Arms:** **T** scores Signpost-D-4B with this harness (it should land near 0.712); **S42** (and S1042, S2042) scores one of its seeds alone, the single-seed baseline; **D1** is the recipe on Qwen3.5-2B; **D2** and **D3** add JEPA with the EMA target (two forms of ẑ, below); **D4** adds JEPA with the frozen 4B as the target encoder (below the training cells); **D5** takes that target from the 4B's middle layers, and its 6-layer transformer predictor reads only the answer position's states from the trained model's layers `STUDENT_LAYERS` (16 to 28), so the answer representation the head reads must itself carry the target. Our dev set (200 records from 8 domains not in training) picks the epoch; the test split (400 records, 2,000 decisions, gold included) is scored once per arm, never used for choices. Readouts on test: the head (servable by vLLM), the pointer, and both averaged.

**Checkpoints** every `CKPT_EVERY` steps and after each epoch, in `CKPT_DIR`; a rerun resumes, and a finished arm's results are read back.
''')

md("## Parameters")
code(r'''
import os

def env(name, default):
    return os.environ.get(name, default)

STUDENT = env("STUDENT", "Qwen/Qwen3.5-2B")
TEACHER = env("TEACHER", "skamalj/signpost-d-4b")     # Signpost-D-4B: 0.712 on the test split
DATASET = "skamalj/signpost-d-decisions"              # our records: train (2,575) and dev (200)
TD = "LocalLLaMA/typed-decisions"                     # the benchmark: the test split is scored, never trained on
ARMS = env("ARMS", "T,D1,D2").split(",")              # T: the 4B; D1: 2B, no JEPA; D2, D3: 2B with JEPA
JEPA_ARMS = {"D2": True, "D3": False}                 # arm -> ẑ includes the answer vector (D2) or not (D3)
LATENT_ARMS = {"D4", "D5"}                            # JEPA against the frozen 4B's latent of the gold answer
TARGET_LAYERS = [int(x) for x in env("TARGET_LAYERS", "12,16,20,24").split(",")]   # D5: the 4B's middle layers
STUDENT_LAYERS = [int(x) for x in env("STUDENT_LAYERS", "16,17,18,19,20,21,22,23,24,25,26,27,28").split(",")]
LAMBDA_LATENT = float(env("LAMBDA_LATENT", "1.0"))
SEED = int(env("SEED", "42"))                         # training: LoRA start, dropout
DATA_SEED = int(env("DATA_SEED", "42"))               # record order and option shuffles: the same across seeds,
                                                      # so D4's teacher latents are shared by every seed
MERGE_SEEDS = env("MERGE_SEEDS", "42,7,1042").split(",")   # arm M4: these D4 seeds merged
MERGE_SCALE = float(env("MERGE_SCALE", "0.75"))       # Signpost-D-4B's merge: mean weight change x 0.75
LORA_RANK = int(env("LORA_RANK", "16"))
LR = float(env("LR", "1e-4"))                         # LoRA weights
LR_PREDICTOR = float(env("LR_PREDICTOR", "3e-4"))     # predictor and temperature (D2)
LR_PREDICTOR_D3 = float(env("LR_PREDICTOR_D3", "1e-3"))   # D3: the predictor carries ẑ alone
EPOCHS = int(env("EPOCHS", "2"))
WARMUP_FRAC = 0.15
LORA_DROPOUT = 0.05
SCORE_DIST_W = 0.5                                    # expected-level term for score questions
CASES_PER_STEP = 2                                    # records per optimizer step (all their questions)
EPOCH_KL_TOL = 0.005                                  # the last epoch is kept unless its dev KL is clearly worse
TRAIN_MAX_TOKENS = 4096                               # a longer state is cut in training
EVAL_MAX_TOKENS = 8191                                # vLLM serves 8,192 tokens including the answer
EVAL_BATCH = 16
PRED_LAYERS, PRED_WIDTH, PRED_HEADS = 6, 384, 6       # the JEPA predictor: a 6-layer transformer (narrow)
LAMBDA_JEPA, LAMBDA_POINT = float(env("LAMBDA_JEPA", "0.5")), float(env("LAMBDA_POINT", "0.5"))
TAU_START = 0.1
EMA_START, EMA_END = 0.996, 0.9999                    # target LoRA momentum, cosine schedule
CKPT_EVERY = 300
CKPT_DIR = env("CKPT_DIR", "ckpt10")
STATE_MODE = env("STATE_MODE", "")                    # oracle check on test: full | no_evidence | derived (appended)
                                                      # or replace_full | replace_no_evidence (instead of the record)
STATES_URL = ("https://raw.githubusercontent.com/skamalj/llms-the-hard-way/decider-jepa/"
              "data/decision_state/v1/td_test.jsonl")
COMPILED_URL = ("https://raw.githubusercontent.com/skamalj/llms-the-hard-way/decider-jepa/"
                "data/compiled_question/v1/td_test.jsonl")   # STATE_MODE=compiled: statement + question
TRAIN_COMPILED = env("TRAIN_COMPILED", "") == "1"   # training questions with a compiled brief read it instead
COMPILED_TRAIN_URL = ("https://raw.githubusercontent.com/skamalj/llms-the-hard-way/decider-jepa/"
                      "data/compiled_question/v1/train_stage2.jsonl")
FINAL_PATH = env("FINAL_PATH", "")                    # arm F: a trained run's final weights (e.g. D1 on the 2B)
os.makedirs(CKPT_DIR, exist_ok=True)
''')

md(r'''
## Packages and the Hugging Face token

The Triton kernels for Qwen3.5's linear-attention layers (`fla-core`, which needs `einops`): without them the model runs on a much slower reference path, so the notebook stops if they do not load. The token reads our private dataset; only where it came from is printed.
''')
code(r'''
!pip install -q einops peft
!pip install -q --no-deps flash-linear-attention fla-core

import copy, gc, json, math, random, statistics, subprocess, sys, time
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "the environment"
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN"), "Kaggle secret"
    except Exception:
        pass
    import getpass
    token = getpass.getpass("Hugging Face token (read access): ").strip()
    return (token, "the prompt") if token else (None, "NOT FOUND")


HF_TOKEN, token_from = hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
print(f"Hugging Face token: {token_from}")
fla_check = subprocess.run([sys.executable, "-c", "from fla.ops.gated_delta_rule import chunk_gated_delta_rule"],
                           capture_output=True, text=True)
print(f"linear-attention kernels: {'fast (fla-core)' if fla_check.returncode == 0 else 'NOT available: slow reference path'}")
assert fla_check.returncode == 0, fla_check.stderr[-800:]   # too slow without them: stop here
''')

md("## Device and log")
code(r'''
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def amp():
    return torch.autocast("cuda", dtype=DTYPE)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def gpu():
    return f"GPU {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB (peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f})"


log(f"torch {torch.__version__} | transformers {transformers.__version__} | {torch.cuda.get_device_name()} | "
    f"{str(DTYPE).split('.')[-1]} | arms {ARMS} | seed {SEED}")
''')

md(r'''
## Questions as items

As Signpost-D renders them (the same as vLLM): true/false as `yes` then `no` with the criteria's descriptions; choice by the option names as written, each with its description; score by its level strings, lowest first. Each question becomes one item: the state (JSON written compactly), the question, its options and the target distribution. Our records and Typed Decisions' records have the same shape.
''')
code(r'''
def render(q):
    criteria = q.get("criteria")
    if q["type"] == "noul":
        criteria = criteria or {}
        return {"yes": criteria.get("true"), "no": criteria.get("false")}, ["true", "false"]
    if q["type"] == "choice":
        return {str(k): None if v is None else str(v) for k, v in criteria.items()}, list(criteria)
    return {str(level): None for level in criteria}, [str(i) for i in range(len(criteria))]


def parsed(x):
    return json.loads(x) if isinstance(x, str) else x


def items_of(case, group=""):
    state = parsed(case["state"])
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    gold_all = parsed(case["gold"])
    out = []
    for qid, q in parsed(case["questions"]).items():
        options, keys = render(q)
        if len(options) != len(keys) or len(options) > 26:
            continue
        gold = gold_all[qid]["probabilities"]
        out.append({"task": "d_" + q["type"], "id": case["id"], "qid": qid, "state": state, "group": group,
                    "question": "" if q.get("instructions") is None else str(q["instructions"]),
                    "options": options, "levels": list(options) if q["type"] == "score" else None,
                    "soft": {name: float(gold[key]) for name, key in zip(options, keys)}})
    return out


def our_split(split):
    path = hf_hub_download(DATASET, f"data/{split}.jsonl", repo_type="dataset", token=HF_TOKEN)
    return [json.loads(line) for line in open(path, encoding="utf-8")]


train_cases, dev_cases = our_split("train"), our_split("dev")
td_path = hf_hub_download(TD, "all/test-00000-of-00001.parquet", repo_type="dataset")
td_cases = pd.read_parquet(td_path).to_dict("records")
train_d = [it for c in train_cases for it in items_of(c)]
dev_d = [it for c in dev_cases for it in items_of(c)]
test_d = [it for c in td_cases for it in items_of(c, c["workflow"])]
if TRAIN_COMPILED:                                    # arm C: train on briefs where they exist
    import urllib.request
    briefs = {}
    for line in urllib.request.urlopen(COMPILED_TRAIN_URL).read().decode("utf-8").splitlines():
        o = json.loads(line)
        briefs[(o["record_id"], o["question_id"])] = (o["statement"], o["question"])
    for it in train_d:
        if (it["id"], it["qid"]) in briefs:
            it["state"], it["question"] = briefs[(it["id"], it["qid"])]
    print(f"training: compiled briefs in place of the record for "
          f"{sum((it['id'], it['qid']) in briefs for it in train_d)} of {len(train_d)} questions")
if STATE_MODE == "compiled":                          # statement + question in place of the record and question
    import urllib.request
    compiled = {}
    for line in urllib.request.urlopen(COMPILED_URL).read().decode("utf-8").splitlines():
        o = json.loads(line)
        compiled[(o["record_id"], o["question_id"])] = (o["statement"], o["question"])
    for it in test_d:
        if (it["id"], it["qid"]) in compiled:
            it["state"], it["question"] = compiled[(it["id"], it["qid"])]
    print(f"compiled questions in place of the record for {sum((it['id'], it['qid']) in compiled for it in test_d)}"
          f" of {len(test_d)} test questions")
elif STATE_MODE:                                      # the oracle check: each test question's decision state
    import urllib.request
    states = {}
    for line in urllib.request.urlopen(STATES_URL).read().decode("utf-8").splitlines():
        o = json.loads(line)
        text = o["state_text"]
        if STATE_MODE.endswith("no_evidence"):        # facts, derived values, discrepancies, missing only
            text = text[:text.find("EVIDENCE")].rstrip() if "EVIDENCE" in text else text
        elif STATE_MODE == "derived":                 # the DERIVED lines only
            text = text[text.find("DERIVED"):text.find("DISCREPANCIES")].rstrip()
        states[(o["record_id"], o["question_id"])] = text
    for it in test_d:
        if (it["id"], it["qid"]) in states:
            it["extra"] = "Decision state:\n" + states[(it["id"], it["qid"])]
            it["replace"] = STATE_MODE.startswith("replace")
    print(f"decision states ({STATE_MODE}) attached to {sum('extra' in it for it in test_d)} of {len(test_d)} test questions")
print(f"train {len(train_cases)} records / {len(train_d)} questions | dev {len(dev_cases)} / {len(dev_d)} | "
      f"Typed Decisions test {len(td_cases)} / {len(test_d)} {dict(Counter(it['task'] for it in test_d))}")
''')

md(r'''
## The prompt and where the options end

vLLM's user turn is the state, a blank line, then `Question: …`, one line per labelled option, and the reply line; no system message; the chat template adds the generation prompt with thinking off. `encode` returns the token ids, the answer position (the last token) and, for the JEPA target, the position of the last token of each option's line, found from the tokenizer's character offsets. A state too long for the budget is cut at its end.
''')
code(r'''
LABELS = {"d_choice": [chr(65 + i) for i in range(26)], "d_noul": ["yes", "no"],
          "d_score": [str(i) for i in range(10)]}
REPLY = {"d_choice": "Answer with the letter of one option only.",
         "d_noul": "Answer with yes or no only.",
         "d_score": "Answer with the number of one level only."}


def question_lines(item):
    """-> (lines of the question text, index of each option's line)."""
    lines = [f"Question: {item['question']}"] if item["question"] else []
    at = []
    for label, (name, description) in zip(LABELS[item["task"]], item["options"].items()):
        if item["task"] == "d_noul":
            line = f"{label}: {description}" if description else label
        elif item["task"] == "d_choice":
            line = f"{label}: {name} - {description}" if description else f"{label}: {name}"
        else:
            line = f"{label}: {name}"
        at.append(len(lines))
        lines.append(line)
    lines.append(REPLY[item["task"]])
    return lines, at


def encode(tok, item, max_tokens):
    """-> (ids, answer position, last-token position of each option line)."""
    for _ in range(3):
        lines, at = question_lines(item)
        qtext = "\n".join(lines)
        if item.get("replace"):                       # the decision state instead of the record
            content = f"{item['extra']}\n\n{qtext}"
        else:
            extra = f"{item['extra']}\n\n" if item.get("extra") else ""
            content = f"{item['state']}\n\n{extra}{qtext}"
        prompt = tok.apply_chat_template([{"role": "user", "content": content}],
                                         add_generation_prompt=True, tokenize=False, enable_thinking=False)
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        if len(ids) <= max_tokens:
            break
        state_ids = tok(item["state"], add_special_tokens=False)["input_ids"]
        keep = max(len(state_ids) - (len(ids) - max_tokens) - 16, 0)
        item = {**item, "state": tok.decode(state_ids[:keep])}
    start = prompt.rindex(qtext)
    line_start = [start + sum(len(l) + 1 for l in lines[:i]) for i in range(len(lines))]
    ends = []
    for i in at:
        last_char = line_start[i] + len(lines[i]) - 1
        inside = [t for t, (a, b) in enumerate(offsets) if a <= last_char < b]
        ends.append(inside[-1] if inside else max(t for t, (a, b) in enumerate(offsets) if a < b <= last_char + 1))
    return ids, len(ids) - 1, ends
''')

md(r'''
## Loading a model and its label tokens

The text model in the working precision, with gradient checkpointing for training. The labels' token ids are found as vLLM finds them: each label must add exactly one token after the prompt's tail. The answer is the hidden state at the last position times the output embeddings of the question's labels.
''')
code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(name, train):
    tok = AutoTokenizer.from_pretrained(name, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, token=HF_TOKEN, dtype=DTYPE, device_map={"": 0})
    m.config.use_cache = False
    if train:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, m


def generation_tail(tok):
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False,
                                    enable_thinking=False)
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    added = set(tok.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(ids) if t in added), default=0)
    return ids[start:]


def label_token_ids(tok):
    tail = generation_tail(tok)
    text = tok.decode(tail)
    out = {}
    for task, labels in LABELS.items():
        ids = []
        for label in labels:
            extended = tok(text + label, add_special_tokens=False)["input_ids"]
            assert extended[:-1] == tail and extended[-1] not in ids, f"label {label!r} is not one distinct token"
            ids.append(extended[-1])
        out[task] = ids
    return out


def inner_of(m):
    base = m.get_base_model() if hasattr(m, "get_base_model") else m
    return base.model.language_model if hasattr(base.model, "language_model") else base.model
''')

md(r'''
## The model

`Decider` holds the model (with LoRA when training), its inner transformer, the output embeddings and the label tokens; with JEPA also the EMA target copy, the predictor and τ. `forward` runs a batch of questions (padded on the right) and returns the head's logits and, with JEPA, the pointer's logits and the pieces of the JEPA loss.
''')
code(r'''
LORA_TARGETS = r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$"


class Predictor(nn.Module):
    """A narrow transformer over the model's whole output, read at the answer position."""

    def __init__(self, hidden, residual, out_dim=None):
        super().__init__()
        self.inp = nn.Linear(hidden, PRED_WIDTH)
        layer = nn.TransformerEncoderLayer(PRED_WIDTH, PRED_HEADS, 4 * PRED_WIDTH, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, PRED_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(PRED_WIDTH)
        self.out = nn.Linear(PRED_WIDTH, out_dim or hidden)
        if residual:                                  # ẑ starts as the answer vector
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, h, att, pos):
        x = self.layers(self.inp(h), src_key_padding_mask=att == 0)
        return self.out(self.norm(x[torch.arange(len(pos), device=h.device), pos]))


class LayerPredictor(nn.Module):
    """D5: a narrow transformer over the answer position's states from STUDENT_LAYERS (one token per layer, with
    a learned layer embedding), mean-pooled to the target's size. It sees nothing but the answer position, so
    meeting the target means the answer representation itself must carry it."""

    def __init__(self, hidden, n_layers, out_dim):
        super().__init__()
        self.inp = nn.Linear(hidden, PRED_WIDTH)
        self.layer_emb = nn.Parameter(torch.zeros(n_layers, PRED_WIDTH))
        layer = nn.TransformerEncoderLayer(PRED_WIDTH, PRED_HEADS, 4 * PRED_WIDTH, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, PRED_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(PRED_WIDTH)
        self.out = nn.Linear(PRED_WIDTH, out_dim)

    def forward(self, stack):                         # [items, n_layers, hidden]
        x = self.layers(self.inp(stack) + self.layer_emb)
        return self.out(self.norm(x.mean(1)))


class Decider:
    def __init__(self, name, train, jepa, residual=True, latent_dim=None, layered=False):
        self.tok, m = load(name, train)
        self.jepa, self.residual, self.latent_dim, self.layered = jepa, residual, latent_dim, layered
        if train:
            from peft import LoraConfig, get_peft_model
            m = get_peft_model(m, LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=LORA_DROPOUT,
                                             target_modules=LORA_TARGETS))
            for p in m.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        self.m = m
        self.inner = inner_of(m)
        self.head_w = m.get_output_embeddings().weight
        self.label_ids = label_token_ids(self.tok)
        hidden = self.head_w.shape[1]
        if jepa:
            self.target = copy.deepcopy(m).requires_grad_(False).eval()
            self.target_inner = inner_of(self.target)
            self.predictor = Predictor(hidden, residual).to(DEVICE)
            self.log_tau = torch.tensor(math.log(TAU_START), device=DEVICE, requires_grad=True)
        if latent_dim:                                # D4, D5: predict the 4B's latent; no EMA copy
            self.predictor = (LayerPredictor(hidden, len(STUDENT_LAYERS), latent_dim) if layered
                              else Predictor(hidden, False, latent_dim)).to(DEVICE)
        n_train = sum(p.numel() for p in self.params()) if train else 0
        log(f"loaded {name} | trainable {n_train / 1e6:.1f} M | {gpu()}")

    def params(self):
        out = [p for p in self.m.parameters() if p.requires_grad]
        if self.jepa:
            out += list(self.predictor.parameters()) + [self.log_tau]
        if self.latent_dim:
            out += list(self.predictor.parameters())
        return out

    def lora_pairs(self):
        return [(pt, ps) for (n, pt), ps in zip(self.target.named_parameters(), self.m.parameters()) if "lora_" in n]

    def train(self, mode=True):
        self.m.train(mode)
        if self.jepa or self.latent_dim:
            self.predictor.train(mode)
        if self.jepa:
            self.target.eval()

    def forward(self, items, max_tokens, pointer, latent=False):
        built = [encode(self.tok, it, max_tokens) for it in items]
        length = max(len(b[0]) for b in built)
        pad = self.tok.pad_token_id
        ids = torch.tensor([b[0] + [pad] * (length - len(b[0])) for b in built], device=DEVICE)
        att = torch.tensor([[1] * len(b[0]) + [0] * (length - len(b[0])) for b in built], device=DEVICE)
        last = torch.tensor([b[1] for b in built], device=DEVICE)
        rows = torch.arange(len(items), device=DEVICE)
        K = max(len(it["options"]) for it in items)
        valid = torch.tensor([[j < len(it["options"]) for j in range(K)] for it in items], device=DEVICE)
        with amp():
            run = self.inner(input_ids=ids, attention_mask=att, output_hidden_states=latent and self.layered)
        h = run.last_hidden_state
        a = h[rows, last]
        label = torch.tensor([self.label_ids[it["task"]][:len(it["options"])] + [0] * (K - len(it["options"]))
                              for it in items], device=DEVICE)
        out = {"head": torch.einsum("nh,nkh->nk", a.float(), self.head_w[label].float()).masked_fill(~valid, -1e4)}
        if pointer:
            ends = torch.tensor([b[2] + [0] * (K - len(b[2])) for b in built], device=DEVICE)
            with torch.no_grad(), amp():
                ht = self.target_inner(input_ids=ids, attention_mask=att).last_hidden_state
            v = ht[rows[:, None], ends].float()                                   # [items, K, hidden]
            w = valid.float().unsqueeze(-1)
            v = (v - (v * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)) * w    # centred per question
            with amp():
                p = self.predictor(h, att, last)
                z = (a + p if self.residual else p).float()
            cos = F.cosine_similarity(z.unsqueeze(1), v, dim=-1)
            out["point"] = (cos / self.log_tau.exp().clamp(min=0.01)).masked_fill(~valid, -1e4)
            out["cos"], out["valid"] = cos, valid
        if latent:
            with amp():
                if self.layered:
                    stack = torch.stack([run.hidden_states[l][rows, last] for l in STUDENT_LAYERS], dim=1)
                    out["z"] = self.predictor(stack).float()
                else:
                    out["z"] = self.predictor(h, att, last).float()
            out["chunks"] = len(TARGET_LAYERS) if self.layered else 1
        return out

    def free(self):
        for k in ("m", "inner", "head_w", "target", "target_inner", "predictor"):
            setattr(self, k, None)
        gc.collect()
        torch.cuda.empty_cache()
''')

md(r'''
## Losses and measures

The head's loss is Signpost-D's: cross-entropy against the gold distribution plus, for score questions, the distance between expected levels. With JEPA: KL for the pointer, and 1 − cos between ẑ and the gold option's centred target, weighted by the gold. The measures are the benchmark's: accuracy (the top option is the gold's top option), KL from the gold, Brier and 10-bin ECE, overall, by type and by workflow.
''')
code(r'''
def gold_matrix(items, K):
    g = torch.tensor([[it["soft"][n] for n in it["options"]] + [0.0] * (K - len(it["options"])) for it in items],
                     device=DEVICE)
    return g / g.sum(1, keepdim=True)


def head_loss(scores, items):
    g = gold_matrix(items, scores.shape[1])
    logp = F.log_softmax(scores, dim=1)
    loss = -(g * logp).sum(1).mean()
    is_score = torch.tensor([it["task"] == "d_score" for it in items], device=DEVICE)
    if is_score.any():
        level = torch.arange(scores.shape[1], device=DEVICE, dtype=torch.float32)
        n = torch.tensor([len(it["options"]) for it in items], device=DEVICE, dtype=torch.float32)
        gap = ((logp.exp() - g) * level).sum(1).abs() / (n - 1).clamp(min=1)
        loss = loss + SCORE_DIST_W * gap[is_score].mean()
    return loss


def losses(out, items):
    parts = {"head": head_loss(out["head"], items)}
    if "point" in out:
        g = gold_matrix(items, out["point"].shape[1])
        parts["point"] = (g * (g.clamp(min=1e-12).log() - F.log_softmax(out["point"], dim=1))).sum(1).mean()
        parts["jepa"] = (g * (1 - out["cos"])).sum(1).mean()
    if "z" in out:
        t = torch.stack([it["latent"] for it in items]).to(DEVICE).float()
        n = out["chunks"]                              # D5: one cosine per target layer, averaged
        z = out["z"].view(len(items), n, -1)
        parts["latent"] = (1 - F.cosine_similarity(z, t.view(len(items), n, -1), dim=-1)).mean()
    total = (parts["head"] + LAMBDA_POINT * parts.get("point", 0) + LAMBDA_JEPA * parts.get("jepa", 0)
             + LAMBDA_LATENT * parts.get("latent", 0))
    return total, {k: v.item() for k, v in parts.items()}


def readouts(out):
    probs = {"head": F.softmax(out["head"], dim=1)}
    if "point" in out:
        probs["point"] = F.softmax(out["point"], dim=1)
        probs["both"] = (probs["head"] + probs["point"]) / 2
    return probs


def metrics(rows):
    """rows: (item, {option: probability})."""
    def summary(rs):
        bins = [[0, 0, 0.0] for _ in range(10)]
        acc, kl, brier = [], [], []
        for it, p in rs:
            g = it["soft"]
            top = max(p, key=p.get)
            ok = top == max(g, key=g.get)
            acc.append(ok)
            kl.append(sum(gv * math.log(gv / max(p[n], 1e-9)) for n, gv in g.items() if gv > 0))
            brier.append(sum((p[n] - g[n]) ** 2 for n in g))
            b = bins[min(int(p[top] * 10), 9)]
            b[0] += 1; b[1] += ok; b[2] += p[top]
        ece = sum(b[0] / len(rs) * abs(b[1] / b[0] - b[2] / b[0]) for b in bins if b[0])
        return {"n": len(rs), "accuracy": round(statistics.mean(acc), 4), "kl": round(statistics.mean(kl), 4),
                "brier": round(statistics.mean(brier), 4), "ece": round(ece, 4)}

    out = {"all": summary(rows)}
    for key in ("task", "group"):
        for v in sorted({it[key] for it, _ in rows if it[key]}):
            out[v] = summary([(it, p) for it, p in rows if it[key] == v])
    return out


@torch.no_grad()
def predict(model, items, pointer):
    """-> {readout: [(item, {option: probability})]}, batches of similar length."""
    model.train(False)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["state"]))
    rows = {}
    for s in range(0, len(order), EVAL_BATCH):
        chunk = [items[i] for i in order[s:s + EVAL_BATCH]]
        for name, probs in readouts(model.forward(chunk, EVAL_MAX_TOKENS, pointer)).items():
            for it, row in zip(chunk, probs.float().cpu().tolist()):
                rows.setdefault(name, []).append((it, {n: row[j] for j, n in enumerate(it["options"])}))
    return rows


def score(model, items, name, pointer):
    result = {r: metrics(rows) for r, rows in predict(model, items, pointer).items()}
    for r, m in result.items():
        groups = " | ".join(f"{k} {v['accuracy']}" for k, v in m.items() if k != "all")
        log(f"{name} [{r}]: {m['all']} | {groups}")
    return result
''')

md(r'''
## Training order, optimizer and EMA

Every epoch sees every training record once, in an order shuffled with the seed and epoch; a record's questions go together, and the options of its choice questions are shuffled (true/false and score keep their order). AdamW (weight decay 0.01): the LoRA weights at `LR`, the predictor and τ at `LR_PREDICTOR`; a linear warm-up over 15% of the steps, then a linear decay to zero. After every step the target's LoRA weights move towards the trained ones (EMA).
''')
code(r'''
def by_case(items):
    groups = []
    for it in items:
        if groups and groups[-1][0]["id"] == it["id"]:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def training_order(epoch):
    cases = by_case(train_d)
    random.Random(DATA_SEED + epoch).shuffle(cases)
    return cases[: len(cases) // CASES_PER_STEP * CASES_PER_STEP]


def prepare(case, epoch):
    rng = random.Random(f"{DATA_SEED}-{epoch}-{case[0]['id']}")
    out = []
    for it in case:
        it = dict(it)
        if it["task"] == "d_choice":
            names = list(it["options"])
            rng.shuffle(names)
            it["options"] = {n: it["options"][n] for n in names}
        out.append(it)
    return out


def make_optimizer(model, total):
    groups = [{"params": [p for p in model.m.parameters() if p.requires_grad], "lr": LR}]
    if model.jepa:
        groups.append({"params": list(model.predictor.parameters()) + [model.log_tau],
                       "lr": LR_PREDICTOR if model.residual else LR_PREDICTOR_D3})
    if model.latent_dim:
        groups.append({"params": list(model.predictor.parameters()), "lr": LR_PREDICTOR_D3})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    return opt, sched


@torch.no_grad()
def ema_update(model, step, total):
    m = EMA_END - (EMA_END - EMA_START) * (math.cos(math.pi * step / total) + 1) / 2
    for pt, ps in model.lora_pairs():
        pt.mul_(m).add_(ps.detach().to(pt.dtype), alpha=1 - m)
    return m
''')

md(r'''
## Checkpoints

A resume point holds the trained weights (LoRA, and with JEPA the target's LoRA, the predictor and τ), the optimizer and schedule, the step and the dev history; written to a temporary file first. A finished arm keeps only its final weights and its results.
''')
code(r'''
def trained_state(model):
    state = {n: p.detach().cpu() for n, p in model.m.named_parameters() if p.requires_grad}
    if model.jepa:
        state.update({"target." + n: p.detach().cpu() for n, p in model.target.named_parameters() if "lora_" in n})
        state.update({"predictor." + n: p.detach().cpu() for n, p in model.predictor.state_dict().items()})
        state["log_tau"] = model.log_tau.detach().cpu()
    if model.latent_dim:
        state.update({"predictor." + n: p.detach().cpu() for n, p in model.predictor.state_dict().items()})
    return state


def load_trained(model, state):
    params = dict(model.m.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            if n.startswith("target."):
                dict(model.target.named_parameters())[n[7:]].copy_(v)
            elif n.startswith("predictor."):
                continue
            elif n == "log_tau":
                model.log_tau.copy_(v)
            else:
                params[n].copy_(v)
        if model.jepa or model.latent_dim:
            model.predictor.load_state_dict({n[10:]: v for n, v in state.items() if n.startswith("predictor.")})


def save_point(path, model, opt, sched, st):
    torch.save({"weights": trained_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(), **st},
               path + ".tmp")
    os.replace(path + ".tmp", path)
    log(f"  checkpoint at step {st['step']}")
''')

md(r'''
## D4's and D5's targets: the 4B's latent of the gold answer

For every training question, in each epoch's option order, the frozen Signpost-D-4B reads vLLM's prompt **with the gold answer's label token appended**; its hidden state there, its representation of the right decision in context, is the target: from the last layer for D4, and for **D5** from the middle layers `TARGET_LAYERS` (12, 16, 20, 24 of 32), each standardised and concatenated. An LLM's last layer is specialised for choosing the next token; its middle layers carry more of the meaning, so D5 uses them. The 2B never sees the answer: its predictor must produce that latent from the prompt alone. The targets are computed once (and saved), then standardised per dimension, so the few very large dimensions of an LLM's hidden state do not decide the cosine. They encode the gold answer, also on the questions where the 4B itself would answer wrongly.
''')
code(r'''
def teacher_latents(layers=None):
    """-> {epoch: {(record id, question id): standardised latent}}: the last layer (D4), or the given layers'
    states concatenated, each standardised (D5)."""
    tag = "" if layers is None else "_L" + "-".join(map(str, layers))
    path = os.path.join(CKPT_DIR, (f"teacher_latents{tag}.pt" if DATA_SEED == 42
                                   else f"teacher_latents{tag}_{DATA_SEED}.pt"))
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    partial = path.replace(".pt", "_raw.pt")   # saved after each epoch, so a restart resumes
    raw = torch.load(partial, weights_only=False) if os.path.exists(partial) else {}
    teacher, t0 = Decider(TEACHER, False, False), time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        if any(k[0] == epoch for k in raw):
            log(f"teacher latents, epoch {epoch}: read back")
            continue
        items = [it for case in training_order(epoch) for it in prepare(case, epoch)]
        items.sort(key=lambda it: len(it["state"]))
        for s in range(0, len(items), EVAL_BATCH):
            chunk = items[s:s + EVAL_BATCH]
            seqs = []
            for it in chunk:
                ids, _, _ = encode(teacher.tok, it, TRAIN_MAX_TOKENS)
                gold = max(range(len(it["options"])), key=lambda j: it["soft"][list(it["options"])[j]])
                seqs.append(ids + [teacher.label_ids[it["task"]][gold]])
            length = max(len(x) for x in seqs)
            pad = teacher.tok.pad_token_id
            ids = torch.tensor([x + [pad] * (length - len(x)) for x in seqs], device=DEVICE)
            att = torch.tensor([[1] * len(x) + [0] * (length - len(x)) for x in seqs], device=DEVICE)
            with torch.no_grad(), amp():
                run = teacher.inner(input_ids=ids, attention_mask=att, output_hidden_states=layers is not None)
            rows = torch.arange(len(seqs), device=DEVICE)
            at = torch.tensor([len(x) - 1 for x in seqs], device=DEVICE)
            if layers is None:
                last = run.last_hidden_state[rows, at]
            else:
                last = torch.cat([run.hidden_states[l][rows, at] for l in layers], dim=-1)
            for it, v in zip(chunk, last.float().cpu()):
                raw[(epoch, it["id"], it["qid"])] = v
        log(f"teacher latents, epoch {epoch}: {sum(1 for k in raw if k[0] == epoch)} | "
            f"{(time.perf_counter() - t0) / 60:.1f} min")
        torch.save(raw, partial + ".tmp")
        os.replace(partial + ".tmp", partial)
    teacher.free()
    stack = torch.stack(list(raw.values()))
    mean, std = stack.mean(0), stack.std(0).clamp(min=1e-3)
    out = {e: {} for e in range(1, EPOCHS + 1)}
    for (e, rid, qid), v in raw.items():
        out[e][(rid, qid)] = ((v - mean) / std).half()
    log(f"teacher latents: dimension {stack.shape[1]} | mean |value| before standardising "
        f"{stack.abs().mean():.2f}, largest dimension {stack.abs().mean(0).max():.1f}")
    torch.save(out, path)
    return out
''')

md(r'''
## Training one arm

Two records per optimizer step (each record's questions in one micro-batch), gradient clipping at 1.0. Every 100 steps: the losses and, with JEPA, cos(ẑ, gold target) against the best other option, how alike a question's (uncentred) targets would be, and τ. After each epoch the dev set is scored (head readout) and the epoch is kept unless its KL is clearly worse than the best.
''')
code(r'''
def train_arm(name, jepa, residual=True, latent=False, layered=False):
    lat = teacher_latents(TARGET_LAYERS if layered else None) if latent else None
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = Decider(STUDENT, True, jepa, residual,
                    latent_dim=next(iter(lat[1].values())).shape[0] if latent else None, layered=layered)
    per_epoch = len(training_order(1)) // CASES_PER_STEP
    total = EPOCHS * per_epoch
    opt, sched = make_optimizer(model, total)
    point = os.path.join(CKPT_DIR, f"{name}.pt")
    st = {"step": 0, "history": [], "best": None}
    if os.path.exists(point):
        saved = torch.load(point, map_location="cpu", weights_only=False)
        load_trained(model, saved["weights"])
        opt.load_state_dict(saved["opt"]); sched.load_state_dict(saved["sched"])
        st = {k: saved[k] for k in ("step", "history", "best")}
        log(f"{name}: resumed at step {st['step']}/{total}")
    log(f"{name}: {per_epoch} steps per epoch, {total} steps | JEPA {jepa}")
    params = model.params()
    t0, recent = time.perf_counter(), []
    for epoch in range(st["step"] // per_epoch + 1, EPOCHS + 1):
        model.train(True)
        order = training_order(epoch)
        for s in range(st["step"] - (epoch - 1) * per_epoch, per_epoch):
            opt.zero_grad(set_to_none=True)
            for a in range(CASES_PER_STEP):
                items = prepare(order[s * CASES_PER_STEP + a], epoch)
                if latent:
                    for it in items:
                        it["latent"] = lat[epoch][(it["id"], it["qid"])]
                out = model.forward(items, TRAIN_MAX_TOKENS, jepa, latent)
                loss, parts = losses(out, items)
                (loss / CASES_PER_STEP).backward()
                recent = (recent + [parts["head"]])[-200:]
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            st["step"] += 1
            m = ema_update(model, st["step"], total) if jepa else 0
            if st["step"] % 100 == 0:
                more = ""
                if jepa:
                    with torch.no_grad():
                        cos, valid = out["cos"], out["valid"]
                        g = gold_matrix(items, cos.shape[1]).argmax(1)
                        rows = torch.arange(len(items), device=DEVICE)
                        other = cos.masked_fill(~valid, -2).scatter(1, g[:, None], -2).max(1).values
                        more = (f" | cos gold {cos[rows, g].mean().item():.3f} best other {other.mean().item():.3f}"
                                f" | tau {model.log_tau.exp().item():.3f} | ema {m:.5f}")
                log(f"{name} epoch {epoch} step {st['step']}/{total} | head loss (last 200) "
                    f"{statistics.mean(recent):.4f} | batch {' '.join(f'{k} {v:.3f}' for k, v in parts.items())} | "
                    f"grad {norm:.2f} | lr {opt.param_groups[0]['lr']:.1e} | "
                    f"{(time.perf_counter() - t0) / 60:.1f} min | {gpu()}{more}")
            if st["step"] % CKPT_EVERY == 0 and st["step"] % per_epoch:
                save_point(point, model, opt, sched, st)
        dev = score(model, dev_d, f"{name} dev, epoch {epoch}", jepa)
        st["history"].append({"epoch": epoch, **{r: m["all"] for r, m in dev.items()}})
        best_kl = min(h["head"]["kl"] for h in st["history"])
        if dev["head"]["all"]["kl"] <= best_kl + EPOCH_KL_TOL:
            st["best"] = {"epoch": epoch, "weights": trained_state(model)}
        save_point(point, model, opt, sched, st)
    if st["best"]["epoch"] != EPOCHS:
        load_trained(model, st["best"]["weights"])
    log(f"{name}: trained in {(time.perf_counter() - t0) / 60:.1f} min | keeping epoch {st['best']['epoch']}")
    return model, st
''')

md(r'''
## Merging seeds (arm M4)

As Signpost-D-4B was made: each D4 seed's weight change (its LoRA product B·A on every adapted projection) is computed, the changes are averaged over `MERGE_SEEDS`, and the mean times `MERGE_SCALE` is added to Qwen3.5-2B's own weights. The merged model is read by its head, as vLLM would serve it.
''')
code(r'''
def merged_model(arm, seeds):
    from peft.tuners.lora import LoraLayer
    model = Decider(STUDENT, True, False)
    layers = {n: m for n, m in model.m.named_modules() if isinstance(m, LoraLayer)}
    params = dict(model.m.named_parameters())
    total = {}
    with torch.no_grad():
        for seed in seeds:
            state = torch.load(os.path.join(CKPT_DIR, f"{arm}-s{seed}_final.pt"), weights_only=False)
            for n, v in state.items():
                if n in params:
                    params[n].copy_(v)
            for n, layer in layers.items():
                d = layer.get_delta_weight("default").float()
                total[n] = d if n not in total else total[n] + d
        for n, layer in layers.items():
            w = layer.base_layer.weight
            w.copy_((w.float() + MERGE_SCALE * total[n] / len(seeds)).to(w.dtype))
            for p in layer.lora_B.parameters():
                p.zero_()                             # the change now lives in the base weights
    log(f"merged {arm} seeds {seeds} at {MERGE_SCALE} into {len(layers)} projections")
    return model
''')

md(r'''
## The arms

**T** scores Signpost-D-4B (no training). **D1** and **D2** train Qwen3.5-2B, then score our dev set and Typed Decisions' test split. A finished arm's results are read back from `CKPT_DIR`.
''')
code(r'''
results = {}
for arm in ARMS:
    name = arm if arm in ("T", "F", "B") or arm.startswith("S") else f"{arm}-s{SEED}"
    done = os.path.join(CKPT_DIR, f"{name}_done.json")
    if os.path.exists(done):
        results[name] = json.load(open(done))
        log(f"{name}: finished earlier, read back")
        continue
    if arm == "T":
        model, st = Decider(TEACHER, False, False), {"history": []}
    elif arm == "B":                                  # the plain base model STUDENT, untrained
        model, st = Decider(STUDENT, False, False), {"history": []}
    elif arm == "F":                                  # a trained run's final weights (FINAL_PATH, base STUDENT)
        model, st = Decider(STUDENT, True, False), {"history": []}
        load_trained(model, torch.load(FINAL_PATH, weights_only=False))
    elif arm.startswith("S"):                         # one Signpost-D-4B seed alone (no JEPA), from its repo
        from peft import PeftModel
        model, st = Decider("Qwen/Qwen3.5-4B", False, False), {"history": []}
        model.m = PeftModel.from_pretrained(model.m, TEACHER, subfolder=f"adapters/seed{arm[1:]}", token=HF_TOKEN)
        model.inner = inner_of(model.m)
    elif arm.startswith("M"):
        name = f"{arm}-{'-'.join(MERGE_SEEDS)}"
        model, st = merged_model("D" + arm[1:], MERGE_SEEDS), {"history": []}
    else:
        model, st = train_arm(name, jepa=arm in JEPA_ARMS, residual=JEPA_ARMS.get(arm, True),
                              latent=arm in LATENT_ARMS, layered=arm == "D5")
    jepa = arm in JEPA_ARMS
    res = {"dev": score(model, dev_d, f"{name} dev", jepa),
           "test": score(model, test_d, f"{name} Typed Decisions test", jepa),
           "history": st["history"]}
    if arm not in ("T", "F", "B") and not arm.startswith(("M", "S")):
        torch.save(trained_state(model), os.path.join(CKPT_DIR, f"{name}_final.pt"))
        if os.path.exists(os.path.join(CKPT_DIR, f"{name}.pt")):
            os.remove(os.path.join(CKPT_DIR, f"{name}.pt"))
    json.dump(res, open(done, "w"), indent=1)
    results[name] = res
    model.free()
    del model
''')

md("## The comparison")
code(r'''
print(f"{'arm':10s} {'readout':8s} {'test acc':>9s} {'KL':>7s} {'Brier':>7s} {'ECE':>7s} {'dev acc':>8s}   by workflow")
for name, res in results.items():
    for r, m in res["test"].items():
        flows = " | ".join(f"{k[:8]} {v['accuracy']:.3f}" for k, v in m.items() if not k.startswith("d_") and k != "all")
        a = m["all"]
        print(f"{name:10s} {r:8s} {a['accuracy']:>9.4f} {a['kl']:>7.4f} {a['brier']:>7.4f} {a['ece']:>7.4f} "
              f"{res['dev'][r]['all']['accuracy']:>8.4f}   {flows}")
print("published: Signpost-D-4B 0.712 on the test split (vLLM, the benchmark's scorer)")
''')

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                                "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for kind, text in cells:
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    nb["cells"].append(cell)
json.dump(nb, open("10_td_jepa_2b.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(nb["cells"]), "cells")
