"""Stage 4: Direct Preference Optimization, written from scratch (no reward model).

L = -log sigmoid( beta [ log pi_theta(y_w|x)/pi_ref(y_w|x) - log pi_theta(y_l|x)/pi_ref(y_l|x) ] ), beta = 0.1,
log pi(y|x) = sum of token log-probs over the summary tokens only.

Usage: python train_dpo.py          (add --limit 64 for a quick smoke test)
"""
import argparse

from torch.utils.data import DataLoader

from common import *

BETA, LR, BATCH_PAIRS, MICRO_PAIRS, CLIP, LOG_EVERY = 0.1, 2e-6, 8, 2, 1.0, 250


def seq_logps(model, ids, am, pl, n):
    """log pi(y|x) = sum over summary tokens -> (chosen (n,), rejected (n,))."""
    m = summary_mask(am, pl).float()
    seq = (token_logprobs(model, ids, am) * m).sum(1)
    return seq[:n], seq[n:]


@torch.no_grad()
def reference_val_logps(reference, loader):
    c, r = [], []
    for b in loader:
        ids, am, pl, n = stack_pair_batch(b)
        pc, pr = seq_logps(reference, ids, am, pl, n)
        c.append(pc); r.append(pr)
    return torch.cat(c), torch.cat(r)


@torch.no_grad()
def validate(policy, loader, ref_c, ref_r):
    """Implicit-reward accuracy: fraction of pairs where the expression inside sigma is positive."""
    was_training = policy.training
    policy.eval()
    i, correct, loss_sum = 0, 0, 0.0
    for b in loader:
        ids, am, pl, n = stack_pair_batch(b)
        pc, pr = seq_logps(policy, ids, am, pl, n)
        z = BETA * ((pc - ref_c[i:i + n]) - (pr - ref_r[i:i + n]))
        correct += int((z > 0).sum().item()); loss_sum += -F.logsigmoid(z).sum().item(); i += n
    policy.train(was_training)
    return {"accuracy": correct / i, "loss": loss_sum / i}


def main(limit=None):
    seed_everything()
    pref_train, pref_val = load_preference_splits(tok)        # the same 20,000 / 1,000 pairs as the reward model
    if limit:
        pref_train, pref_val = pref_train[:limit], pref_val[:max(8, limit // 4)]
    policy = enable_gradient_checkpointing(load_policy(SFT_DIR, frozen=False)).to(DEVICE).train()
    reference = freeze(load_policy(SFT_DIR, frozen=True))
    assert all(not p.requires_grad for p in reference.parameters())
    print("DPO reference frozen: requires_grad=False for every parameter.")
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    scaler = make_scaler()
    collate = partial(collate_pref, pad_id=PAD_ID)
    train_loader = DataLoader(pref_train, BATCH_PAIRS, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(pref_val, 4, collate_fn=collate)
    ref_c, ref_r = reference_val_logps(reference, val_loader)       # reference is frozen -> compute once

    hist = {"step": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_implicit_acc": []}
    run_loss = run_acc = 0.0
    run_steps, start = 0, time.time()
    for step, b in enumerate(train_loader, 1):
        ids, am, pl, n = stack_pair_batch(b)
        opt.zero_grad(set_to_none=True)
        step_loss = step_acc = 0.0
        for s in range(0, n, MICRO_PAIRS):
            sid, sam, spl, m = select_pairs(ids, am, pl, n, s, min(s + MICRO_PAIRS, n))
            pc, pr = seq_logps(policy, sid, sam, spl, m)
            with torch.no_grad():
                rc, rr = seq_logps(reference, sid, sam, spl, m)
            z = BETA * ((pc - rc) - (pr - rr))
            loss = -F.logsigmoid(z).sum() / n
            if step == 1 and s == 0:     # correctness check: pi_theta == pi_ref at the first step -> loss = log 2
                first = -F.logsigmoid(z).mean().item()
                assert abs(first - math.log(2)) < 1e-3, first
                print(f"DPO first-step check passed: loss = {first:.4f} (log 2 = {math.log(2):.4f})")
            scaler.scale(loss).backward()
            step_loss += loss.item(); step_acc += (z > 0).float().sum().item() / n
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
        scaler.step(opt); scaler.update()
        run_loss += step_loss; run_acc += step_acc; run_steps += 1
        if step % LOG_EVERY == 0 or step == len(train_loader):
            val = validate(policy, val_loader, ref_c, ref_r)
            for k, v in zip(hist, (step, run_loss / run_steps, run_acc / run_steps, val["loss"], val["accuracy"])):
                hist[k].append(v)
            print(f"DPO step {step:5d} | train loss {run_loss / run_steps:.4f} | val loss {val['loss']:.4f} | "
                  f"val implicit-reward acc {val['accuracy']:.4f}", flush=True)
            run_loss = run_acc = 0.0; run_steps = 0
    seconds = time.time() - start
    print(f"DPO time: {seconds / 60:.1f} min")

    policy.config.use_cache = True
    policy.save_pretrained(DPO_DIR, safe_serialization=True)
    tok.save_pretrained(DPO_DIR)
    save_json(RESULTS_DIR / "dpo_history.json", {**hist, "final_implicit_acc": hist["val_implicit_acc"][-1], "seconds": seconds})

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["step"], hist["train_loss"], marker="o"); ax[0].set_title("DPO training loss")
    ax[1].plot(hist["step"], hist["val_implicit_acc"], marker="o"); ax[1].axhline(0.5, ls="--", c="gray")
    ax[1].set_title("validation implicit-reward accuracy")
    for a in ax:
        a.set_xlabel("optimizer step"); a.grid(True)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "fig4_dpo.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only N training pairs (smoke test)")
    main(ap.parse_args().limit)
