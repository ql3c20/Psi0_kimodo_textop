#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-ilWc5D/yzh/Psi0}"

TASK="${TASK:-G1WholebodyXMovePickTeleop-v0}"
LEVELS="${LEVELS:-level-0 level-1 level-2}"
TARGET_SUCCESSES_PER_LEVEL="${TARGET_SUCCESSES_PER_LEVEL:-33}"
MAX_ATTEMPTS_PER_LEVEL="${MAX_ATTEMPTS_PER_LEVEL:-80}"
EVAL_GPU="${EVAL_GPU:-4}"
HOST="${HOST:-localhost}"
PORT="${PORT:-22085}"
SAVE_ATTEMPT_VIDEOS="${SAVE_ATTEMPT_VIDEOS:-1}"
RESEED_DR_LEVEL_AFTER_FIRST_PASS="${RESEED_DR_LEVEL_AFTER_FIRST_PASS:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${PSI0_ROOT}/data/simple/${TASK}-eval-fullstate-33x3-${STAMP}}"

for LEVEL in $LEVELS; do
  echo "=============================="
  echo "Collecting ${TASK} ${LEVEL}"
  echo "Output root: ${SAVE_ROOT}"
  echo "=============================="

  LEVEL="$LEVEL" \
  TARGET_SUCCESSES="$TARGET_SUCCESSES_PER_LEVEL" \
  MAX_ATTEMPTS="$MAX_ATTEMPTS_PER_LEVEL" \
  EVAL_GPU="$EVAL_GPU" \
  HOST="$HOST" \
  PORT="$PORT" \
  SAVE_ATTEMPT_VIDEOS="$SAVE_ATTEMPT_VIDEOS" \
  RESEED_DR_LEVEL_AFTER_FIRST_PASS="$RESEED_DR_LEVEL_AFTER_FIRST_PASS" \
  SAVE_ROOT="$SAVE_ROOT" \
  bash "${PSI0_ROOT}/scripts/deploy/simple_eval_to_lerobot_collect.sh"
done
