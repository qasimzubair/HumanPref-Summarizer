#!/usr/bin/env bash
# Runs every stage in order and keeps the logs (they contain the correctness-check output required in the README).
set -euo pipefail
mkdir -p results/logs
python starter/check_starter.py 2>&1 | tee results/logs/check_starter.log
python train_sft.py             2>&1 | tee results/logs/sft.log
python train_reward.py          2>&1 | tee results/logs/reward.log
python train_ppo.py             2>&1 | tee results/logs/ppo.log
python train_dpo.py             2>&1 | tee results/logs/dpo.log
python evaluate.py              2>&1 | tee results/logs/evaluate.log
