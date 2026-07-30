#!/usr/bin/env bash
set -euo pipefail

: "${VERL_DIR:?Set VERL_DIR to an upstream VERL checkout.}"
: "${MODEL_PATH:?Set MODEL_PATH to the base or SFT selector checkpoint.}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the RL training parquet file.}"
: "${VAL_DATA:?Set VAL_DATA to the RL validation parquet file.}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REWARD_FILE="${REPO_DIR}/src/deepsearch/training/verl_reward.py"

cd "${VERL_DIR}"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  custom_reward_function.path="${REWARD_FILE}" \
  custom_reward_function.name=compute_score \
  "$@"
