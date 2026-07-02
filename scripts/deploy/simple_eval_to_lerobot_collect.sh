#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-ilWc5D/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"

TASK="${TASK:-G1WholebodyXMovePickTeleop-v0}"
LEVEL="${LEVEL:-level-0}"
HOST="${HOST:-localhost}"
PORT="${PORT:-22085}"
EVAL_GPU="${EVAL_GPU:-4}"
TARGET_SUCCESSES="${TARGET_SUCCESSES:-33}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-80}"
EPISODE_START="${EPISODE_START:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
SAVE_ATTEMPT_VIDEOS="${SAVE_ATTEMPT_VIDEOS:-1}"
RESEED_DR_LEVEL_AFTER_FIRST_PASS="${RESEED_DR_LEVEL_AFTER_FIRST_PASS:-}"
FORCE_DR_LEVEL="${FORCE_DR_LEVEL:-}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-${PSI0_ROOT}/data/simple/${TASK}-eval-fullstate-${LEVEL}-${STAMP}}"

cd "$SIMPLE_ROOT"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNI_KIT_ACCEPT_EULA=Y
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"

CMD=(
  python -m simple.cli.eval_psi0_to_lerobot
  "simple/${TASK}"
  psi0_decoupled_wbc
  --host="$HOST"
  --port="$PORT"
  --data-dir="data/evals/simple-eval/${TASK}/${LEVEL}"
  --save-dir="$SAVE_ROOT"
  --sim-mode=mujoco_isaac
  --headless
  --target-successes="$TARGET_SUCCESSES"
  --max-attempts="$MAX_ATTEMPTS"
  --episode-start="$EPISODE_START"
  --overwrite
)

if [[ -n "$MAX_EPISODE_STEPS" ]]; then
  CMD+=(--max-episode-steps="$MAX_EPISODE_STEPS")
fi

if [[ "$SAVE_ATTEMPT_VIDEOS" == "0" ]]; then
  CMD+=(--no-save-attempt-videos)
fi

if [[ -n "$RESEED_DR_LEVEL_AFTER_FIRST_PASS" ]]; then
  CMD+=(--reseed-dr-level-after-first-pass="$RESEED_DR_LEVEL_AFTER_FIRST_PASS")
fi

if [[ -n "$FORCE_DR_LEVEL" ]]; then
  CMD+=(--force-dr-level="$FORCE_DR_LEVEL")
fi

if [[ "${EXPORT_POLICY_ACTION:-0}" == "1" ]]; then
  CMD+=(--export-policy-action)
fi

printf 'Running collection for %s %s -> %s\n' "$TASK" "$LEVEL" "$SAVE_ROOT"
"${CMD[@]}"
