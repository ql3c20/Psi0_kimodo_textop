#!/usr/bin/env bash
set -euo pipefail

# Sequentially evaluate both SONIC-trained checkpoint-10000 chains on GPUs 4-7.

COMMAND="${1:-all}"
PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
HUMANOID_ROOT="${HUMANOID_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena}"
ROT_WRAPPER="${PSI0_ROOT}/scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh"
NATIVE_WRAPPER="${PSI0_ROOT}/scripts/deploy/humanoidarena_football_gr00t_n17_sonic_eval.sh"
ENV_CONFIG_YAML="${HUMANOID_ROOT}/isaaclab_twist2_g1/tasks/common_test_config/base_test/football_single_sonic78_train_range_goalframe_only.yaml"

ROT_CHECKPOINT="${ROT_CHECKPOINT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-football-sonic-gr00t-n17-rot6d59-v2-prefixrtc-delay0to12-8gpu-bs512-step10000/checkpoint-10000}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-sonic-native-v2-8gpu-bs512-step10000/checkpoint-10000}"
ROT_RESULTS_DIR="${ROT_RESULTS_DIR:-${PSI0_ROOT}/evals/humanoidarena_football_sonic_rot6d59_v2_ckpt10000_goalframe_only_eval50-allvideo}"
NATIVE_RESULTS_DIR="${NATIVE_RESULTS_DIR:-${PSI0_ROOT}/evals/humanoidarena_football_sonic_native_v2_ckpt10000_goalframe_only_eval50-allvideo}"

common_env=(
  EVAL_SEEDS="0 1 2 3 4"
  REPEATS_PER_SEED=10
  MAX_STEPS=1300
  VIDEO_FPS=50
  PERSISTENT_SIM=1
  RECORD_VIDEO_EVERY_N=1
  ENV_CONFIG_YAML="$ENV_CONFIG_YAML"
)

run_rot() {
  env "${common_env[@]}" \
    ARENA_TASK_KIND=football ARENA_EVAL_PROFILE=goalframe_only \
    GR00T_MODEL_PATH="$ROT_CHECKPOINT" RESULTS_DIR="$ROT_RESULTS_DIR" \
    SERVE_GPU=4 EVAL_GPU=5 KIMODO_GPU=6 \
    bash "$ROT_WRAPPER" "${1:-all}" football
}

run_native() {
  env "${common_env[@]}" \
    GR00T_MODEL_PATH="$NATIVE_CHECKPOINT" RESULTS_DIR="$NATIVE_RESULTS_DIR" \
    SERVER_GPU=6 EVAL_GPU=7 \
    bash "$NATIVE_WRAPPER" "${1:-all}"
}

case "$COMMAND" in
  all)
    run_rot all
    run_native all
    ;;
  rot) run_rot all ;;
  native) run_native all ;;
  dry-run)
    run_rot dry-run
    run_native dry-run
    ;;
  *) echo "Usage: bash $0 {all|rot|native|dry-run}" >&2; exit 2 ;;
esac
