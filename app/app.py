"""Gradio front end: paste a Reddit post, see SFT / PPO / DPO summaries side by side, each with its reward-model score.

Run:  CHECKPOINTS=/path/to/checkpoints python app/app.py
Expects checkpoints/{sft,ppo,dpo}/ (HF format) and checkpoints/reward/reward_model.pt
"""
import os
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(os.environ.get("CHECKPOINTS", "checkpoints"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PROMPT_TOKENS, MAX_NEW_TOKENS = 448, 48

tok = AutoTokenizer.from_pretrained(ROOT / "sft")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
PAD_ID, EOS_ID = tok.pad_token_id, tok.eos_token_id


class RewardModel(nn.Module):           # same layout as in the training notebook
    def __init__(self, sft_dir):
        super().__init__()
        lm = AutoModelForCausalLM.from_pretrained(sft_dir, torch_dtype=torch.float32)
        self.backbone, self.reward_head = lm.model, nn.Linear(lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        pos_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos_ids, use_cache=False).last_hidden_state
        pos = torch.arange(h.size(1), device=h.device).unsqueeze(0).expand_as(attention_mask)
        last = pos.masked_fill(attention_mask == 0, -1).max(1).values
        return self.reward_head(h[torch.arange(h.size(0)), last].float()).squeeze(-1)


def load_lm(name):
    return AutoModelForCausalLM.from_pretrained(ROOT / name, torch_dtype=torch.float32).to(DEVICE).eval()


MODELS = {name: load_lm(name) for name in ("sft", "ppo", "dpo")}
RM = RewardModel(ROOT / "sft")
_state = torch.load(ROOT / "reward" / "reward_model.pt", map_location="cpu", weights_only=False)
RM.load_state_dict(_state.get("model_state_dict", _state))     # notebook checkpoint or plain state dict
RM = RM.to(DEVICE).eval()


@torch.no_grad()
def summarize(post: str):
    post = post.split("TL;DR:")[0].strip()
    prompt_ids = tok(post + "\nTL;DR:", add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        msg = f"Post is {len(prompt_ids)} tokens; the models were trained on posts up to {MAX_PROMPT_TOKENS}. Please shorten it."
        return [msg, "", "", "", "", ""]
    ids = torch.tensor([prompt_ids], device=DEVICE)
    mask = torch.ones_like(ids)
    out = []
    for name in ("sft", "ppo", "dpo"):
        gen = MODELS[name].generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=PAD_ID, eos_token_id=EOS_ID)[0, ids.shape[1]:].tolist()
        if EOS_ID in gen:
            gen = gen[: gen.index(EOS_ID) + 1]
        full = torch.tensor([prompt_ids + gen], device=DEVICE)
        score = RM(full, torch.ones_like(full)).item()
        out += [tok.decode([t for t in gen if t != EOS_ID]).strip(), f"{score:+.3f}"]
    return out


with gr.Blocks(title="Learning to Summarize") as demo:
    gr.Markdown("## SFT vs PPO vs DPO summaries\nPaste a Reddit post (subreddit, title and body).")
    post = gr.Textbox(lines=12, label="Post")
    btn = gr.Button("Summarize")
    with gr.Row():
        boxes = []
        for name in ("SFT", "PPO", "DPO"):
            with gr.Column():
                boxes += [gr.Textbox(label=f"{name} summary", lines=4), gr.Textbox(label=f"{name} reward-model score")]
    btn.click(summarize, post, boxes)

if __name__ == "__main__":
    demo.launch()
