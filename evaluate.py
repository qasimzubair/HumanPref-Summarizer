"""Final evaluation: compare SFT, PPO and DPO on the 300 held-out evaluation posts.

Everything is computed on the exact generated token ids (prompt ids + generated ids, EOS included when generated),
which is the format the reward model was trained on. Nothing is re-tokenised from stripped text.

Usage:
  python evaluate.py            # generate, score, Tables 2-3, eval_summaries.jsonl, samples.md, human-eval sheet
  python evaluate.py --tally    # after BOTH members filled results/human_eval_sheet.csv: Table 4 + human_eval.csv
"""
import argparse

import pandas as pd

from common import *
from train_reward import load_reward_model

METHODS = ["sft", "ppo", "dpo"]
CKPTS = {"sft": SFT_DIR, "ppo": PPO_DIR, "dpo": DPO_DIR}


@torch.no_grad()
def reward_scores(rm, prompt_lists, resp_lists, batch_size=16):
    out = []
    for s in range(0, len(prompt_lists), batch_size):
        seqs = [p + r for p, r in zip(prompt_lists[s:s + batch_size], resp_lists[s:s + batch_size])]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long); am = torch.zeros_like(ids)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        with autocast_context():
            out.extend(rm(ids.to(DEVICE), am.to(DEVICE)).float().cpu().tolist())
    return out


@torch.no_grad()
def summary_logps(model, prompt_lists, resp_lists, batch_size=8):
    """sum_t log pi(y_t | x, y_<t) of the given summary ids."""
    out = []
    for s in range(0, len(prompt_lists), batch_size):
        ps, rs = prompt_lists[s:s + batch_size], resp_lists[s:s + batch_size]
        seqs = [p + r for p, r in zip(ps, rs)]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long); am = torch.zeros_like(ids)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        ids, am = ids.to(DEVICE), am.to(DEVICE)
        m = summary_mask(am, torch.tensor([len(p) for p in ps])).float()
        out.extend((token_logprobs(model, ids, am) * m).sum(1).cpu().tolist())
    return out


def build_table2():
    h = {k: json.load(open(RESULTS_DIR / f"{k}_history.json")) for k in ("sft", "reward", "ppo", "dpo")}
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    t = pd.DataFrame([
        {"Stage": "SFT", "Final train loss": h["sft"]["train_loss"][-1], "Validation metric": f"val loss {h['sft']['val_loss'][-1]:.4f}", "Time (min)": h["sft"]["seconds"] / 60, "GPU": gpu},
        {"Stage": "Reward model", "Final train loss": h["reward"]["train_loss"][-1], "Validation metric": f"pairwise acc {h['reward']['final_accuracy']:.4f}", "Time (min)": h["reward"]["seconds"] / 60, "GPU": gpu},
        {"Stage": "PPO", "Final train loss": float(np.mean(h["ppo"]["policy_loss"][-10:])), "Validation metric": f"final mean r_phi {np.mean(h['ppo']['reward'][-10:]):.3f} (last 10 it.)", "Time (min)": h["ppo"]["seconds"] / 60, "GPU": gpu},
        {"Stage": "DPO", "Final train loss": h["dpo"]["train_loss"][-1], "Validation metric": f"implicit reward acc {h['dpo']['final_implicit_acc']:.4f}", "Time (min)": h["dpo"]["seconds"] / 60, "GPU": gpu},
    ])
    print(t.round(4).to_string(index=False))
    t.to_csv(RESULTS_DIR / "table2_training.csv", index=False)


def make_human_sheet(posts, gen_text):
    """30 posts (fixed seed), three summaries in random order, model names hidden."""
    rng = random.Random(SEED)
    idx = rng.sample(range(len(posts)), 30)
    sheet, key = [], {}
    for pid in idx:
        order = METHODS[:]
        rng.shuffle(order)
        key[str(pid)] = dict(zip("ABC", order))
        sheet.append({"post_id": pid, "post": posts[pid]["prompt"],
                      **{f"summary_{l}": gen_text[m][pid] for l, m in zip("ABC", order)},
                      "member1_pick": "", "member2_pick": ""})
    pd.DataFrame(sheet).to_csv(RESULTS_DIR / "human_eval_sheet.csv", index=False)      # give this to the raters
    save_json(RESULTS_DIR / "human_eval_key.json", key)                                # keep hidden until both have voted
    print("Wrote results/human_eval_sheet.csv (fill member1_pick / member2_pick with A, B or C) and human_eval_key.json")


def tally():
    df = pd.read_csv(RESULTS_DIR / "human_eval_sheet.csv")
    key = json.load(open(RESULTS_DIR / "human_eval_key.json"))
    for who in ("member1", "member2"):
        df[f"{who}_method"] = [key[str(p)][str(c).strip().upper()] for p, c in zip(df["post_id"], df[f"{who}_pick"])]
    df["agree"] = df["member1_method"] == df["member2_method"]
    df.to_csv(RESULTS_DIR / "human_eval.csv", index=False)
    rows = {w: df[f"{w}_method"].value_counts().reindex(METHODS, fill_value=0).to_dict() for w in ("member1", "member2")}
    print("Table 4:", rows, f"| agreement {int(df['agree'].sum())} / {len(df)}")


def write_samples(posts, gen_text, rewards, n=10):
    """Draft of results/samples.md: biggest reward gains and drops vs SFT. The 'why' lines are for you to write."""
    delta = {m: np.array(rewards[m]) - np.array(rewards["sft"]) for m in ("ppo", "dpo")}
    ranked = []                      # best and worst vs SFT for each method, round by round, until n unique posts
    for k in range(n):
        for m in ("ppo", "dpo"):
            ranked += [int(np.argsort(-delta[m])[k]), int(np.argsort(delta[m])[k])]
    picks = list(dict.fromkeys(ranked))[:n]
    lines = ["# Qualitative samples\n"]
    for k, i in enumerate(picks, 1):
        lines += [f"## Sample {k} (eval post {i})", f"**Post (tail):** {posts[i]['prompt'][-400:]}", "",
                  f"- **Human reference:** {posts[i]['summary']}"]
        lines += [f"- **{m.upper()}** (r_phi {rewards[m][i]:+.2f}): {gen_text[m][i]}" for m in METHODS]
        lines += ["- **Clearly better / worse than SFT, and why:** _TODO (write this yourself)_", ""]
    (RESULTS_DIR / "samples.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    from rouge_score import rouge_scorer
    seed_everything()
    posts = load_eval_posts(tok)                              # 300 posts, never trained on
    prompts = [p["prompt_ids"] for p in posts]
    rm = load_reward_model()

    gen_ids, kl, own_logp = {}, {}, {}
    for m in METHODS:
        model = load_policy(CKPTS[m])
        gen_ids[m] = generate_ids(model, posts)
        own_logp[m] = summary_logps(model, prompts, gen_ids[m])
        del model; free_memory()
    sft = load_policy(SFT_DIR)
    for m in METHODS:   # KL to SFT on the method's OWN greedy summary: sum_t [log pi_m - log pi_sft]
        kl[m] = [a - b for a, b in zip(own_logp[m], summary_logps(sft, prompts, gen_ids[m]))]
    del sft; free_memory()

    gen_text = {m: [to_text(x) for x in gen_ids[m]] for m in METHODS}
    rewards = {m: reward_scores(rm, prompts, gen_ids[m]) for m in METHODS}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = {m: [scorer.score(p["summary"].strip(), h)["rougeL"].fmeasure for p, h in zip(posts, gen_text[m])] for m in METHODS}

    sft_r = np.array(rewards["sft"])
    rows = [{"Policy": m.upper(), "Mean r_phi": float(np.mean(rewards[m])),
             "Win rate vs SFT": "-" if m == "sft" else float((np.array(rewards[m]) > sft_r).mean()),
             "KL to SFT": float(np.mean(kl[m])), "Length": float(np.mean([len(x) for x in gen_ids[m]])),
             "ROUGE-L": float(np.mean(rouge_l[m]))} for m in METHODS]
    table3 = pd.DataFrame(rows)
    print(table3.round(4).to_string(index=False))
    table3.to_csv(RESULTS_DIR / "table3_final_comparison.csv", index=False)

    with open(RESULTS_DIR / "eval_summaries.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(posts):
            row = {"id": i, "prompt": p["prompt"], "reference": p["summary"].strip()}
            for m in METHODS:
                row.update({f"{m}_summary": gen_text[m][i], f"{m}_reward": rewards[m][i], f"{m}_kl": kl[m][i],
                            f"{m}_length": len(gen_ids[m][i]), f"{m}_rougeL": rouge_l[m][i]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_samples(posts, gen_text, rewards)
    make_human_sheet(posts, gen_text)
    try:
        build_table2()
    except FileNotFoundError as e:      # e.g. training stages were skipped and no *_history.json exists
        print(f"Skipping Table 2 (missing {Path(e.filename).name}); run the training scripts or add the history files.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally", action="store_true", help="compute Table 4 from the filled human-eval sheet")
    tally() if ap.parse_args().tally else main()
