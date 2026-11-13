"""Stage 2: reward model. Transformer body of the SFT model + nn.Linear(hidden, 1) on the
hidden state of the last REAL token of prompt + summary. Loss: Bradley-Terry.

Usage: python train_reward.py          (add --limit 64 for a quick smoke test)
"""
import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from common import *

LR, BATCH_PAIRS, MICRO_PAIRS, CLIP, LOG_EVERY = 1e-5, 8, 1, 1.0, 250


class RewardModel(nn.Module):
    def __init__(self, sft_dir=SFT_DIR):
        super().__init__()
        lm = load_causal_lm(str(sft_dir), "cpu")
        self.backbone = lm.model                    # transformer body; LM head dropped
        self.hidden_size = lm.config.hidden_size
        self.reward_head = nn.Linear(self.hidden_size, 1)
        del lm

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=make_position_ids(attention_mask), use_cache=False).last_hidden_state
        # last REAL token, found from the attention mask (never from input_ids != pad_token_id)
        pos = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0).expand_as(attention_mask)
        last = pos.masked_fill(attention_mask == 0, -1).max(dim=1).values
        h = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        with torch.autocast(device_type=hidden.device.type, enabled=False):   # head in fp32
            return self.reward_head(h.float()).squeeze(-1)


def load_reward_model(path=REWARD_DIR):
    rm = RewardModel(SFT_DIR)
    state = torch.load(Path(path) / "reward_model.pt", map_location="cpu", weights_only=False)
    rm.load_state_dict(state.get("model_state_dict", state))      # accepts a plain state dict or a {"model_state_dict": ...} checkpoint
    return freeze(rm.to(DEVICE))


@torch.no_grad()
def pair_rewards(model, loader):
    """(r_chosen, r_rejected) arrays for every pair of the loader (in order)."""
    was_training = model.training
    model.eval()
    rc, rr = [], []
    for b in loader:
        ids, am, _, n = stack_pair_batch(b)
        with autocast_context():
            r = model(ids, am).float()
        rc.append(r[:n].cpu()); rr.append(r[n:].cpu())
    model.train(was_training)
    return torch.cat(rc).numpy(), torch.cat(rr).numpy()


def evaluate(model, loader):
    rc, rr = pair_rewards(model, loader)
    loss = float(np.mean(np.log1p(np.exp(-(rc - rr)))))      # -log sigmoid(rc - rr)
    return {"loss": loss, "accuracy": float((rc > rr).mean())}


def report_mistakes(model, pref_val, loader, k=3):
    rc, rr = pair_rewards(model, loader)
    wrong = np.where(rr >= rc)[0]
    wrong = wrong[np.argsort(-(rr[wrong] - rc[wrong]))]       # most confidently wrong first
    lc = np.array([len(r["chosen_ids"]) for r in pref_val]); lr_ = np.array([len(r["rejected_ids"]) for r in pref_val])
    lines = [f"# Reward-model mistakes\n\nAccuracy {(rc > rr).mean():.4f} on {len(rc)} validation pairs; {len(wrong)} mistakes.",
             f"Mean summary length chosen / rejected: {lc.mean():.1f} / {lr_.mean():.1f} tokens. "
             f"Accuracy when chosen is longer: {(rc > rr)[lc > lr_].mean():.3f}; when rejected is longer: {(rc > rr)[lr_ > lc].mean():.3f}\n"]
    for i in wrong[:k]:
        r = pref_val[int(i)]
        lines += [f"## Validation pair {int(i)}", f"**Post (tail):** {r['prompt'][-300:]}", "",
                  f"- chosen   (r={rc[i]:+.2f}): {r['chosen']}", f"- rejected (r={rr[i]:+.2f}): {r['rejected']}",
                  "- Why the model got it wrong: _TODO (write this yourself)_", ""]
    text = "\n".join(lines)
    (RESULTS_DIR / "rm_mistakes.md").write_text(text, encoding="utf-8")
    print(text)


def main(limit=None):
    seed_everything()
    pref_train, pref_val = load_preference_splits(tok)
    if limit:
        pref_train, pref_val = pref_train[:limit], pref_val[:max(8, limit // 4)]
    model = RewardModel(SFT_DIR)
    model.backbone.config.use_cache = False
    try:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.backbone.gradient_checkpointing_enable()
    model = model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = make_scaler()
    collate = partial(collate_pref, pad_id=PAD_ID)
    train_loader = DataLoader(pref_train, BATCH_PAIRS, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(pref_val, 16, collate_fn=collate)

    hist = {"step": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    run_loss, run_pairs, start = 0.0, 0, time.time()
    for step, b in enumerate(train_loader, 1):
        ids, am, pl, n = stack_pair_batch(b)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for s in range(0, n, MICRO_PAIRS):
            sid, sam, _, m = select_pairs(ids, am, pl, n, s, min(s + MICRO_PAIRS, n))
            with autocast_context():
                r = model(sid, sam).float()
            loss = -F.logsigmoid(r[:m] - r[m:]).sum() / n         # Bradley-Terry, mean over the n pairs
            scaler.scale(loss).backward()
            step_loss += loss.item() * n
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(opt); scaler.update()
        run_loss += step_loss; run_pairs += n
        if step % LOG_EVERY == 0 or step == len(train_loader):
            val = evaluate(model, val_loader)
            for k, v in zip(hist, (step, run_loss / run_pairs, val["loss"], val["accuracy"])):
                hist[k].append(v)
            print(f"Reward step {step:5d} | train loss {run_loss / run_pairs:.4f} | val loss {val['loss']:.4f} | "
                  f"val acc {val['accuracy']:.4f}", flush=True)
            run_loss, run_pairs = 0.0, 0
    seconds = time.time() - start
    print(f"Final pairwise accuracy on {len(pref_val)} validation pairs: {hist['val_accuracy'][-1]:.4f} | time {seconds / 60:.1f} min")

    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), REWARD_DIR / "reward_model.pt")        # weights only
    save_json(RESULTS_DIR / "reward_history.json", {**hist, "final_accuracy": hist["val_accuracy"][-1], "seconds": seconds})
    report_mistakes(model, pref_val, val_loader)                          # error analysis

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["step"], hist["train_loss"], marker="o", label="train")
    ax[0].plot(hist["step"], hist["val_loss"], marker="s", label="validation")
    ax[0].set_xlabel("optimizer step"); ax[0].set_ylabel("Bradley-Terry loss"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(hist["step"], hist["val_accuracy"], marker="o"); ax[1].axhline(0.5, ls="--", c="gray")
    ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("validation pairwise accuracy"); ax[1].grid(True)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "fig2_reward_model.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only N training pairs (smoke test)")
    main(ap.parse_args().limit)
