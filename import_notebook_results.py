"""Rebuild results/{sft,reward,ppo,dpo}_history.json from the training state stored inside the checkpoints.

Needed only if the models were trained in the notebook: `evaluate.py` reads these files to build Table 2.
Usage:  python import_notebook_results.py
"""
import torch

from common import CKPT_DIR, RESULTS_DIR, save_json

load = lambda p: torch.load(p, map_location="cpu", weights_only=False)

sft = load(CKPT_DIR / "sft" / "training_state.pt")
save_json(RESULTS_DIR / "sft_history.json", {**sft["history"], "seconds": sft["seconds"]})

rm = load(CKPT_DIR / "reward" / "reward_model.pt")
save_json(RESULTS_DIR / "reward_history.json", {**rm["history"], "final_accuracy": rm["metrics"]["accuracy"], "seconds": rm["seconds"]})

ppo = load(CKPT_DIR / "ppo" / "training_state.pt")
save_json(RESULTS_DIR / "ppo_history.json", {**ppo["history"], "seconds": ppo["seconds"]})

dpo = load(CKPT_DIR / "dpo" / "training_state.pt")
save_json(RESULTS_DIR / "dpo_history.json", {**dpo["history"], "final_implicit_acc": dpo["history"]["val_implicit_acc"][-1], "seconds": dpo["seconds"]})

print("Wrote sft/reward/ppo/dpo _history.json to", RESULTS_DIR)
