"""Shared helpers used by every stage: token_logprobs, summary_mask, batching glue, generation."""
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))           # the given starter files, unchanged

from config import *                                 # noqa: E402,F401,F403  (MODEL_NAME, SEED, ...)
from models import load_tokenizer, load_causal_lm    # noqa: E402
from data import load_sft_splits, load_preference_splits, load_eval_posts   # noqa: E402,F401
from batching import collate_sft, collate_pref, collate_prompts             # noqa: E402,F401

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
SFT_DIR, REWARD_DIR = CKPT_DIR / "sft", CKPT_DIR / "reward"
PPO_DIR, DPO_DIR = CKPT_DIR / "ppo", CKPT_DIR / "dpo"
for _d in (CKPT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

tok = load_tokenizer()
PAD_ID, EOS_ID = tok.pad_token_id, tok.eos_token_id      # NOTE: PAD == EOS here, so never use `ids != PAD_ID` as a mask


# ---------------------------------------------------------------- generic utilities
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

def enable_gradient_checkpointing(model):
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    return model

def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    assert all(not p.requires_grad for p in model.parameters())
    return model

def load_policy(path, frozen=True):
    m = load_causal_lm(str(path), str(DEVICE))
    m.config.pad_token_id = PAD_ID
    return freeze(m) if frozen else m

def free_memory():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def chunk_slices(n, parts):
    size = math.ceil(n / parts)
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]

def make_position_ids(attention_mask):
    """Position of each real token among the real tokens (correct for left AND right padding)."""
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


# ---------------------------------------------------------------- log-prob and mask helpers
def token_logprobs(model, input_ids, attention_mask):
    """log pi(x_t | x_<t) for t = 1..L-1  ->  (B, L-1). Gradients flow if the model is in grad mode."""
    with autocast_context():
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=make_position_ids(attention_mask), use_cache=False).logits
    logits = logits[:, :-1].float()
    target = logits.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return target - torch.logsumexp(logits, dim=-1)

def summary_mask(attention_mask, prompt_lens):
    """(B, L-1) mask, 1 exactly where the TARGET token t (1..L-1) is a summary token:
    t >= prompt_len and t is a real (non-padding) token. Assumes right-padded sequences."""
    L = attention_mask.shape[1]
    t = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)
    pl = prompt_lens.to(attention_mask.device).unsqueeze(1)
    return ((t >= pl) & attention_mask[:, 1:].bool()).long()

def mask_check(records, n=3):
    """Decoding the tokens where summary_mask == 1 must give exactly the summary + EOS."""
    batch = collate_sft(records[:n], PAD_ID)
    m = summary_mask(batch["attention_mask"], batch["prompt_lens"])
    for i in range(n):
        decoded = tok.decode(batch["input_ids"][i, 1:][m[i].bool()])
        print(f"mask==1 tokens of example {i}: {decoded!r}")
        assert decoded == records[i]["summary"] + tok.eos_token, "summary mask is wrong"
    print("SFT mask check passed.")


# ---------------------------------------------------------------- preference-pair batching
def stack_pair_batch(b):
    """collate_pref output -> chosen rows first, then rejected rows, same length (right-padded).
    Returns ids, mask, prompt_lens (all 2n rows, on DEVICE) and n."""
    n = b["chosen_ids"].size(0)
    L = max(b["chosen_ids"].size(1), b["rejected_ids"].size(1))
    pad = lambda x, v: F.pad(x, (0, L - x.size(1)), value=v)
    ids = torch.cat([pad(b["chosen_ids"], PAD_ID), pad(b["rejected_ids"], PAD_ID)])
    mask = torch.cat([pad(b["chosen_mask"], 0), pad(b["rejected_mask"], 0)])
    pl = torch.cat([b["prompt_lens"], b["prompt_lens"]])
    return ids.to(DEVICE), mask.to(DEVICE), pl.to(DEVICE), n

def select_pairs(ids, mask, pl, n, s, e):
    """Pairs s..e-1 of a stacked batch (still chosen first, then rejected), padding trimmed."""
    idx = torch.tensor(list(range(s, e)) + list(range(n + s, n + e)), device=ids.device)
    sub_mask = mask[idx]
    Lm = int(sub_mask.sum(1).max())
    return ids[idx][:, :Lm], sub_mask[:, :Lm], pl[idx], e - s


# ---------------------------------------------------------------- generation
def cut_at_eos(ids):
    ids = list(ids)
    return ids[:ids.index(EOS_ID) + 1] if EOS_ID in ids else ids

def to_text(ids):
    return tok.decode([t for t in ids if t != EOS_ID]).strip()

@torch.no_grad()
def generate_ids(model, records, batch_size=8):
    """Greedy summaries (<= 48 new tokens) for records with 'prompt_ids'. Returns token-id lists,
    cut after the first EOS (EOS included when generated). Prompts are left-padded."""
    model.eval()
    out = []
    for s in range(0, len(records), batch_size):
        b = collate_prompts(records[s:s + batch_size], PAD_ID)
        ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_SUMMARY_TOKENS,
                             do_sample=False, num_beams=1, pad_token_id=PAD_ID, eos_token_id=EOS_ID, use_cache=True)
        out.extend(cut_at_eos(row) for row in gen[:, ids.shape[1]:].tolist())
    return out
