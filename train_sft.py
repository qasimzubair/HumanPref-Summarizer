"""Dataset statistics, summary-mask check, and Stage 1: supervised fine-tuning.

Loss: mean negative log-likelihood of the summary tokens only, over the whole 16-sequence batch.
Usage: python train_sft.py            (add --limit 64 for a quick smoke test)
"""
import argparse

from torch.utils.data import DataLoader

from common import *

LR, BATCH, MICRO, CLIP, LOG_EVERY = 5e-5, 16, 2, 1.0, 250


def table1(sft_train, sft_val, pref_train, pref_val, eval_posts):
    p = [len(r["prompt_ids"]) for r in sft_train]
    s = [len(r["summary_ids"]) for r in sft_train]        # includes the EOS token
    t = {"sft_train": len(sft_train), "sft_val": len(sft_val), "pref_train": len(pref_train),
         "pref_val": len(pref_val), "eval_posts": len(eval_posts),
         "prompt_len_mean": float(np.mean(p)), "prompt_len_max": int(np.max(p)),
         "summary_len_mean": float(np.mean(s)), "summary_len_max": int(np.max(s))}
    print(json.dumps(t, indent=2))
    save_json(RESULTS_DIR / "table1_data.json", t)


@torch.no_grad()
def validation_loss(model, loader):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0.0
    for b in loader:
        ids, am, pl = (b[k].to(DEVICE) for k in ("input_ids", "attention_mask", "prompt_lens"))
        m = summary_mask(am, pl).float()
        total += -(token_logprobs(model, ids, am) * m).sum().item()
        count += m.sum().item()
    model.train(was_training)
    return total / count


def sample_summaries(model, posts, label):
    ids = generate_ids(model, posts)
    return [f"### {label} | post {i}\n{p['prompt'][-300:]}\n\n**Reference:** {p['summary']}\n\n**{label}:** {to_text(g)}\n"
            for i, (p, g) in enumerate(zip(posts, ids))]


def main(limit=None):
    seed_everything()
    sft_train, sft_val = load_sft_splits(tok)
    pref_train, pref_val = load_preference_splits(tok)
    eval_posts = load_eval_posts(tok)
    table1(sft_train, sft_val, pref_train, pref_val, eval_posts)       # dataset statistics
    mask_check(sft_train)                                              # sanity check
    if limit:
        sft_train, sft_val = sft_train[:limit], sft_val[:max(8, limit // 4)]

    # summaries of 5 evaluation posts BEFORE training
    posts5 = eval_posts[:5]
    base = load_policy(MODEL_NAME, frozen=True)
    before = sample_summaries(base, posts5, "BASE (before SFT)")
    del base; free_memory()

    model = enable_gradient_checkpointing(load_policy(MODEL_NAME, frozen=False))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = make_scaler()
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(sft_train, BATCH, shuffle=True, collate_fn=partial(collate_sft, pad_id=PAD_ID), generator=g)
    val_loader = DataLoader(sft_val, 8, collate_fn=partial(collate_sft, pad_id=PAD_ID))

    hist = {"step": [], "train_loss": [], "val_loss": []}
    run_loss, run_steps, start = 0.0, 0, time.time()
    for step, b in enumerate(train_loader, 1):
        ids, am, pl = (b[k].to(DEVICE) for k in ("input_ids", "attention_mask", "prompt_lens"))
        full_mask = summary_mask(am, pl).float()
        n_tokens = full_mask.sum()                  # summary tokens in the whole batch of 16
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for s in range(0, ids.size(0), MICRO):
            sl = slice(s, s + MICRO)
            Lm = int(am[sl].sum(1).max())
            lp = token_logprobs(model, ids[sl, :Lm], am[sl, :Lm])
            loss = -(lp * full_mask[sl, :Lm - 1]).sum() / n_tokens
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        scaler.step(opt); scaler.update()
        run_loss += step_loss; run_steps += 1
        if step % LOG_EVERY == 0 or step == len(train_loader):
            val = validation_loss(model, val_loader)
            hist["step"].append(step); hist["train_loss"].append(run_loss / run_steps); hist["val_loss"].append(val)
            print(f"SFT step {step:5d} | train loss {run_loss / run_steps:.4f} | val loss {val:.4f}", flush=True)
            run_loss, run_steps = 0.0, 0
    seconds = time.time() - start
    print(f"SFT time: {seconds / 60:.1f} min")

    model.config.use_cache = True
    model.save_pretrained(SFT_DIR, safe_serialization=True)
    tok.save_pretrained(SFT_DIR)
    save_json(RESULTS_DIR / "sft_history.json", {**hist, "seconds": seconds})

    model.eval()
    after = sample_summaries(model, posts5, "SFT (after)")
    (RESULTS_DIR / "sft_before_after.md").write_text("\n".join(before + after), encoding="utf-8")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(hist["step"], hist["train_loss"], marker="o", label="train (avg of last 250 steps)")
    plt.plot(hist["step"], hist["val_loss"], marker="s", label="validation")
    plt.xlabel("optimizer step"); plt.ylabel("NLL per summary token"); plt.title("SFT loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig1_sft_loss.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only N training pairs (smoke test)")
    main(ap.parse_args().limit)
