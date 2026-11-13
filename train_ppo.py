"""Stage 3: Proximal Policy Optimization, written from scratch.

Policy pi_theta + value head V_theta (init: SFT), frozen reference (SFT), frozen reward model.
Per iteration: rollout 16 prompts -> store log pi_old, log pi_ref, V_old, r_phi (no grad) ->
per-token reward -> Monte-Carlo returns (gamma=1) and normalised advantages -> K=4 epochs of
full-batch updates (the 16 sequences are processed in two halves whose gradients are summed
before each optimizer.step()).

Usage: python train_ppo.py          (add --iterations 3 for a quick smoke test)
"""
import argparse

import torch.nn as nn

from common import *
from train_reward import load_reward_model

ITERATIONS, PROMPTS, EPOCHS, SPLITS = 200, 16, 4, 2
LR, KL_COEF, CLIP_EPS, VALUE_COEF, ENTROPY_COEF, GAMMA, MAX_GRAD_NORM = 1e-5, 0.05, 0.2, 0.1, 0.01, 1.0, 1.0


def policy_forward(model, value_head, full_ids, full_mask, P, resp, with_value=False):
    """Per generated token a_t: log pi(a_t|s_t), entropy of pi(.|s_t), optionally V(s_t).
    All read from the position just BEFORE a_t (the one whose logits predict a_t)."""
    T = resp.shape[1]
    with autocast_context():
        hidden = model.model(input_ids=full_ids, attention_mask=full_mask,
                             position_ids=make_position_ids(full_mask), use_cache=False).last_hidden_state
    h = hidden[:, P - 1: P - 1 + T].float()
    logits = F.linear(h, model.lm_head.weight)                 # fp32 head: no fp16 noise in log-probs
    lse = torch.logsumexp(logits, dim=-1)
    token_lp = logits.gather(-1, resp.unsqueeze(-1)).squeeze(-1) - lse
    entropy = lse - (torch.softmax(logits, dim=-1) * logits).sum(-1)
    values = value_head(h).squeeze(-1) if with_value else None
    return token_lp, entropy, values


@torch.no_grad()
def rollout(policy, records):
    """Sample one summary per prompt (T=1, no top-k/top-p, <= 48 tokens). A summary ends at its first EOS."""
    policy.eval()
    b = collate_prompts(records, PAD_ID)                        # LEFT-padded prompts
    ids, mask = b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)
    P = ids.shape[1]
    policy.config.use_cache = True
    gen = policy.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_SUMMARY_TOKENS, do_sample=True,
                          temperature=1.0, top_k=0, top_p=1.0, pad_token_id=PAD_ID, eos_token_id=EOS_ID, use_cache=True)
    policy.config.use_cache = False
    resp = gen[:, P:]
    is_eos = resp.eq(EOS_ID).long()
    resp_mask = ((is_eos.cumsum(1) - is_eos) == 0).long()       # keep tokens up to and incl. the first EOS
    return {"P": P, "resp": resp, "resp_mask": resp_mask,
            "full_ids": torch.cat([ids, resp], 1), "full_mask": torch.cat([mask, resp_mask], 1)}


@torch.no_grad()
def collect(policy, value_head, reference, reward_model, roll):
    B, T = roll["resp"].shape
    old_lp = torch.zeros(B, T, device=DEVICE); ref_lp = torch.zeros_like(old_lp)
    old_v = torch.zeros_like(old_lp); scores = torch.zeros(B, device=DEVICE)
    policy.eval()
    for sl in chunk_slices(B, SPLITS):                          # same shapes as the update -> identical numerics
        args = (roll["full_ids"][sl], roll["full_mask"][sl], roll["P"], roll["resp"][sl])
        old_lp[sl], _, old_v[sl] = policy_forward(policy, value_head, *args, with_value=True)
        ref_lp[sl], _, _ = policy_forward(reference, None, *args)
        with autocast_context():
            scores[sl] = reward_model(roll["full_ids"][sl], roll["full_mask"][sl]).float()   # reads last REAL token
    m = roll["resp_mask"].float()
    old_lp, ref_lp, old_v = old_lp * m, ref_lp * m, old_v * m

    # r_t = -beta (log pi_old - log pi_ref) + 1[t = T] r_phi(x, y)
    rewards = -KL_COEF * (old_lp - ref_lp) * m
    last = roll["resp_mask"].sum(1) - 1
    rewards[torch.arange(B, device=DEVICE), last] += scores

    # R_t = sum_{k>=0} gamma^k r_{t+k};  A_t = R_t - V_old(s_t), normalised over all generated tokens
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=DEVICE)
    for t in reversed(range(T)):
        running = rewards[:, t] + GAMMA * running
        returns[:, t] = running
    returns = returns * m
    adv = (returns - old_v) * m
    valid = adv[m.bool()]
    adv = (adv - valid.mean()) / (valid.std(unbiased=False) + 1e-8) * m
    return {"old_lp": old_lp, "old_v": old_v, "scores": scores, "returns": returns, "adv": adv,
            "seq_kl": ((old_lp - ref_lp) * m).sum(1)}             # sum over the summary's tokens


def update(policy, value_head, opt, scaler, roll, st):
    params = list(policy.parameters()) + list(value_head.parameters())
    m_all = roll["resp_mask"].float()
    N = m_all.sum()                                              # every expectation = mean over generated tokens of the batch
    B = m_all.shape[0]
    log = {"pg": [], "vf": [], "ent": [], "clip": [], "dev": []}
    policy.train()
    for epoch in range(EPOCHS):
        opt.zero_grad(set_to_none=True)
        pg_e = vf_e = en_e = clip_e = dev_e = 0.0
        for sl in chunk_slices(B, SPLITS):
            new_lp, ent, v = policy_forward(policy, value_head, roll["full_ids"][sl], roll["full_mask"][sl],
                                            roll["P"], roll["resp"][sl], with_value=True)
            m, a = m_all[sl], st["adv"][sl]
            ratio = torch.exp(new_lp - st["old_lp"][sl])
            surr = torch.minimum(ratio * a, ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * a)
            pg = -(surr * m).sum() / N                           # -L_PPO
            vf = (((v - st["returns"][sl]) ** 2) * m).sum() / N
            en = (ent * m).sum() / N
            loss = pg + VALUE_COEF * vf - ENTROPY_COEF * en
            scaler.scale(loss).backward()                        # halves add up to the full-batch mean
            with torch.no_grad():
                pg_e += pg.item(); vf_e += vf.item(); en_e += en.item()
                clip_e += ((((ratio - 1).abs() > CLIP_EPS).float()) * m).sum().item() / N.item()
                dev_e = max(dev_e, ((ratio - 1).abs() * m).max().item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
        scaler.step(opt); scaler.update()
        for k, val in zip(log, (pg_e, vf_e, en_e, clip_e, dev_e)):
            log[k].append(val)
        if epoch == 0:   # correctness check: in the first epoch r_t = 1 for every token and the clip fraction is 0
            assert dev_e < 2e-2 and clip_e == 0.0, f"epoch-0 max |r-1| = {dev_e:.2e}, clip fraction = {clip_e}"
    return {"policy_loss": float(np.mean(log["pg"])), "value_loss": float(np.mean(log["vf"])),
            "entropy": float(np.mean(log["ent"])), "clip_fraction": float(np.mean(log["clip"])),
            "epoch0_max_ratio_dev": log["dev"][0]}


def main(iterations=ITERATIONS):
    seed_everything()
    sft_train, _ = load_sft_splits(tok)
    order = list(range(len(sft_train)))
    random.Random(SEED).shuffle(order)
    prompts = [sft_train[i] for i in order[:iterations * PROMPTS]]      # 3,200 distinct prompts (prompts only are used)
    assert len(prompts) == iterations * PROMPTS

    policy = enable_gradient_checkpointing(load_policy(SFT_DIR, frozen=False)).to(DEVICE)
    value_head = nn.Linear(policy.config.hidden_size, 1).to(DEVICE)      # new linear head on the last hidden layer
    reference = freeze(load_policy(SFT_DIR, frozen=True))
    reward_model = load_reward_model()
    # correctness check: frozen models
    assert all(not p.requires_grad for p in reference.parameters())
    assert all(not p.requires_grad for p in reward_model.parameters())
    print("Frozen-model check passed: reference and reward model have requires_grad=False.")

    opt = torch.optim.AdamW(list(policy.parameters()) + list(value_head.parameters()), lr=LR)
    scaler = make_scaler()
    keys = ("iteration", "reward", "kl", "length", "clip_fraction", "policy_loss", "value_loss", "entropy", "epoch0_max_ratio_dev")
    hist = {k: [] for k in keys}
    start = time.time()
    for it in range(1, iterations + 1):
        roll = rollout(policy, prompts[(it - 1) * PROMPTS: it * PROMPTS])
        st = collect(policy, value_head, reference, reward_model, roll)
        if it == 1:      # correctness check: policy == SFT == reference, so KL must be 0
            kl0 = st["seq_kl"].mean().item()
            assert abs(kl0) < 1e-2, f"first-iteration KL = {kl0}"
            print(f"PPO first-iteration check passed: KL to reference = {kl0:.6f}")
        upd = update(policy, value_head, opt, scaler, roll, st)
        row = {"iteration": it, "reward": st["scores"].mean().item(), "kl": st["seq_kl"].mean().item(),
               "length": roll["resp_mask"].sum(1).float().mean().item(), **upd}
        for k in keys:
            hist[k].append(row[k])
        if it == 1 or it % 10 == 0:
            print(f"PPO it {it:3d}/{iterations} | r_phi {row['reward']:7.3f} | KL {row['kl']:7.3f} | len {row['length']:5.1f} | "
                  f"clip {row['clip_fraction']:.3f} | pg {row['policy_loss']:.3f} vf {row['value_loss']:.3f} ent {row['entropy']:.3f}", flush=True)
        del roll, st
    seconds = time.time() - start
    print(f"PPO time: {seconds / 60:.1f} min | max epoch-0 |r-1| over all iterations: {max(hist['epoch0_max_ratio_dev']):.2e}")

    policy.config.use_cache = True
    policy.save_pretrained(PPO_DIR, safe_serialization=True)             # policy only, WITHOUT the value head
    tok.save_pretrained(PPO_DIR)
    torch.save(value_head.state_dict(), PPO_DIR / "value_head.pt")
    save_json(RESULTS_DIR / "ppo_history.json", {**hist, "seconds": seconds})

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, key, title in zip(ax.ravel(), ("reward", "kl", "length", "clip_fraction"),
                             ("mean r_phi", "mean KL to reference (sum over summary)", "mean summary length (tokens)", "clip fraction")):
        a.plot(hist["iteration"], hist[key]); a.set_title(title); a.set_xlabel("PPO iteration"); a.grid(True)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "fig3_ppo.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=ITERATIONS, help="PPO iterations (default 200; use 3 for a smoke test)")
    main(ap.parse_args().iterations)
