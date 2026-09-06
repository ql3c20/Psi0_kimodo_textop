#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/yzh/Psi0_kimodo_textop"
TEXTOP_MEMO="${ROOT}/scripts/deploy/eval_scq_simple_task6_textop.sh"
SONIC_MEMO="${ROOT}/scripts/deploy/eval_scq_simple_task6_sonic.sh"
TASK5_TEXTOP_MEMO="${ROOT}/scripts/deploy/eval_scq_simple_task5_textop.sh"
TEXTOP_LAUNCHER="${ROOT}/scripts/deploy/launch_eval_scq_simple_task6_textop.sh"
SONIC_LAUNCHER="${ROOT}/scripts/deploy/launch_eval_scq_simple_task6_sonic.sh"
TASK5_TEXTOP_LAUNCHER="${ROOT}/scripts/deploy/launch_eval_scq_simple_task5_textop.sh"
STATE_DIR="$(mktemp -d /tmp/task6_textop_then_sonic.XXXXXX)"
ACTIVE_PROTOCOL=""

memo_scalar() {
  local memo="$1" name="$2"
  awk -F= -v name="$name" '$1 == name { value = substr($0, index($0, "=") + 1) } END { print value }' "$memo" \
    | sed -E 's/[[:space:]\\]+$//; s/^"//; s/"$//'
}

line_count() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l <"$path"
  else
    echo 0
  fi
}

port_pids() {
  local protocol="$1" port="$2"
  fuser -n "$protocol" "$port" 2>/dev/null || true
}

stop_active_services() {
  local protocol="${1:-$ACTIVE_PROTOCOL}" pids="" port
  [[ -n "$protocol" ]] || return 0
  if [[ "$protocol" == "textop" ]]; then
    for port in 22085 22185; do
      pids+=" $(port_pids tcp "$port")"
    done
  elif [[ "$protocol" == "sonic" ]]; then
    pids+=" $(port_pids tcp 22095)"
  fi
  pids+=" $(port_pids udp 23331)"
  pids="$(tr ' ' '\n' <<<"$pids" | awk 'NF && !seen[$1]++ { printf "%s ", $1 }')"
  if [[ -n "$pids" ]]; then
    echo "[sequence] stopping ${protocol} service PIDs: ${pids}"
    kill $pids 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    if [[ "$protocol" == "textop" ]]; then
      [[ -z "$(port_pids tcp 22085)$(port_pids tcp 22185)$(port_pids udp 23331)" ]] && break
    else
      [[ -z "$(port_pids tcp 22095)$(port_pids udp 23331)" ]] && break
    fi
    sleep 1
  done
  if [[ "$protocol" == "textop" ]]; then
    [[ -z "$(port_pids tcp 22085)$(port_pids tcp 22185)$(port_pids udp 23331)" ]] || {
      echo "TextOp services did not stop cleanly." >&2
      return 1
    }
  else
    [[ -z "$(port_pids tcp 22095)$(port_pids udp 23331)" ]] || {
      echo "SONIC services did not stop cleanly." >&2
      return 1
    }
  fi
  ACTIVE_PROTOCOL=""
}

wait_for_done() {
  local done_file="$1" label="$2" deadline=$((SECONDS + 21600))
  echo "[sequence] waiting for ${label} evaluation to finish..."
  until [[ -f "$done_file" ]]; do
    (( SECONDS < deadline )) || { echo "Timed out waiting for ${label} evaluation." >&2; return 1; }
    sleep 5
  done
  local status
  status="$(<"$done_file")"
  [[ "$status" == "0" ]] || { echo "${label} evaluation exited with status ${status}." >&2; return 1; }
}

validate_results() {
  local eval_dir="$1" initial_lines="$2" expected="$3" label="$4"
  local results="${eval_dir}/results.jsonl" final_lines added
  [[ -f "$results" ]] || { echo "${label} did not create ${results}" >&2; return 1; }
  [[ -f "${eval_dir}/summary.json" ]] || { echo "${label} did not create ${eval_dir}/summary.json" >&2; return 1; }
  final_lines="$(line_count "$results")"
  added=$((final_lines - initial_lines))
  (( added >= expected )) || {
    echo "${label} produced ${added}/${expected} planned result rows." >&2
    return 1
  }
  echo "[sequence] ${label} complete: ${added} new result rows"
  echo "[sequence] results: ${eval_dir}"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_active_services "$ACTIVE_PROTOCOL" || true
  rm -rf -- "$STATE_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

[[ -n "${DISPLAY:-}" ]] || { echo "DISPLAY is not set; run this from the Ubuntu desktop session." >&2; exit 1; }

textop_tag="$(memo_scalar "$TEXTOP_MEMO" RUN_TAG)"
textop_expected="$(memo_scalar "$TEXTOP_MEMO" NUM_EPISODES)"
textop_eval_dir="${ROOT}/third_party/SIMPLE/data/evals_task6_scq_isaac_${textop_tag}"
textop_initial="$(line_count "${textop_eval_dir}/results.jsonl")"
textop_done="${STATE_DIR}/textop.done"

TASK6_KEEP_TERMINALS_OPEN=0 TASK6_EVAL_DONE_FILE="$textop_done" \
  bash "$TEXTOP_LAUNCHER"
ACTIVE_PROTOCOL="textop"
wait_for_done "$textop_done" "TextOp"
validate_results "$textop_eval_dir" "$textop_initial" "$textop_expected" "TextOp"
stop_active_services textop

sonic_tag="$(memo_scalar "$SONIC_MEMO" RUN_TAG)"
sonic_expected="$(memo_scalar "$SONIC_MEMO" NUM_EPISODES)"
sonic_eval_dir="${ROOT}/third_party/SIMPLE/data/evals_task6_scq_isaac_${sonic_tag}"
sonic_initial="$(line_count "${sonic_eval_dir}/results.jsonl")"
sonic_done="${STATE_DIR}/sonic.done"

TASK6_KEEP_TERMINALS_OPEN=0 TASK6_EVAL_DONE_FILE="$sonic_done" \
  bash "$SONIC_LAUNCHER"
ACTIVE_PROTOCOL="sonic"
wait_for_done "$sonic_done" "SONIC"
validate_results "$sonic_eval_dir" "$sonic_initial" "$sonic_expected" "SONIC"
stop_active_services sonic

task5_textop_tag="$(memo_scalar "$TASK5_TEXTOP_MEMO" RUN_TAG)"
task5_textop_expected="$(memo_scalar "$TASK5_TEXTOP_MEMO" NUM_EPISODES)"
task5_textop_eval_dir="${ROOT}/third_party/SIMPLE/data/evals_task5_scq_isaac_${task5_textop_tag}"
task5_textop_initial="$(line_count "${task5_textop_eval_dir}/results.jsonl")"
task5_textop_done="${STATE_DIR}/task5_textop.done"

TASK6_KEEP_TERMINALS_OPEN=0 TASK6_EVAL_DONE_FILE="$task5_textop_done" \
  bash "$TASK5_TEXTOP_LAUNCHER"
ACTIVE_PROTOCOL="textop"
wait_for_done "$task5_textop_done" "Task5 TextOp"
validate_results "$task5_textop_eval_dir" "$task5_textop_initial" "$task5_textop_expected" "Task5 TextOp"
stop_active_services textop

trap - EXIT INT TERM
rm -rf -- "$STATE_DIR"
echo "[sequence] Task6 TextOp, Task6 SONIC, and Task5 TextOp evaluations all completed successfully."
