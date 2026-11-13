# Analysis

Findings from the final runs. Figures, tables and logs referenced below are in this folder.

## 1. Reward-model accuracy and error patterns
_Pairwise accuracy, three representative mistakes (`rm_mistakes.md`) and what they have in common._

## 2. PPO: reward, length and summary quality
_Mean r_phi and KL against iteration (`fig3_ppo.png`); length change; whether a higher reward-model score means a better summary (`human_eval.csv`)._

## 3. The role of the KL coefficient
_What beta_KL does in the KL-constrained objective, and what to expect with beta_KL = 0._

## 4. Clip fraction and update size
_Clip fraction over training and what it says about the size of each policy update._

## 5. DPO without a reward model
_Where the reward goes, why the partition function Z(x) cancels, and implicit-reward accuracy versus reward-model accuracy on the same 1,000 validation pairs._

## 6. Cost and recommendation
_Models held in GPU memory for PPO and for DPO, wall-clock time per stage, and which method to choose for a new project._
