#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/yzh/Psi0_kimodo_textop"
MEMO="${ROOT}/scripts/deploy/eval_scq_simple_task6_textop.sh"
SELF="${ROOT}/scripts/deploy/launch_eval_scq_simple_task6_textop.sh"

extract_block() {
  local target="$1"
  awk -v target="$target" '
    /^```bash[[:space:]]*$/ { block += 1; inside = (block == target); next }
    /^```[[:space:]]*$/ { if (inside) exit; inside = 0; next }
    inside { print }
  ' "$MEMO"
}

wait_http() {
  local url="$1" label="$2" deadline=$((SECONDS + 600))
  until curl --noproxy '*' -fsS "$url" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || { echo "Timed out waiting for ${label}: ${url}" >&2; return 1; }
    sleep 2
  done
  echo "[ready] ${label}: ${url}"
}

wait_udp() {
  local port="$1" deadline=$((SECONDS + 600))
  until ss -H -lun | awk '{print $4}' | grep -Eq "(^|:)${port}$"; do
    (( SECONDS < deadline )) || { echo "Timed out waiting for UDP ${port}" >&2; return 1; }
    sleep 2
  done
  echo "[ready] Isaac UDP ${port}"
}

run_block() {
  local block="$1" command status=0
  command="$(extract_block "$block")"
  [[ -n "$command" ]] || { echo "Missing bash block ${block} in ${MEMO}" >&2; return 1; }
  if [[ "$block" == "4" ]]; then
    wait_http "http://127.0.0.1:22085/health" "GR00T" || status=$?
    [[ "$status" != "0" ]] || wait_http "http://127.0.0.1:22185/config" "Kimodo" || status=$?
    [[ "$status" != "0" ]] || wait_udp 23331 || status=$?
  fi
  [[ "$status" != "0" ]] || eval "$command" || status=$?
  if [[ "$block" == "4" && -n "${TASK6_EVAL_DONE_FILE:-}" ]]; then
    printf '%s\n' "$status" >"${TASK6_EVAL_DONE_FILE}.tmp"
    mv "${TASK6_EVAL_DONE_FILE}.tmp" "$TASK6_EVAL_DONE_FILE"
  fi
  echo
  echo "[terminal] command exited with status ${status}."
  if [[ "${TASK6_KEEP_TERMINALS_OPEN:-1}" == "1" ]]; then
    echo "[terminal] keeping this terminal open."
    exec bash -i
  fi
  return "$status"
}

case "${1:-}" in
  --run-block) run_block "${2:?missing block number}" ;;
  "")
    [[ -n "${DISPLAY:-}" ]] || { echo "DISPLAY is not set; run this from the Ubuntu desktop session." >&2; exit 1; }
    command -v gnome-terminal >/dev/null || { echo "gnome-terminal is required" >&2; exit 1; }
    for port in 22085 22185; do
      if ss -H -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
        echo "TCP ${port} is already occupied; stop the old service first." >&2
        exit 1
      fi
    done
    if ss -H -lun | awk '{print $4}' | grep -Eq '(^|:)23331$'; then
      echo "UDP 23331 is already occupied; stop the old Isaac viewer first." >&2
      exit 1
    fi
    rm -f /dev/shm/simple_task6_isaac_ego.frame
    gnome-terminal --window --title="Task6 TextOp - GR00T" -- bash "$SELF" --run-block 1
    gnome-terminal --window --title="Task6 TextOp - Kimodo" -- bash "$SELF" --run-block 2
    gnome-terminal --window --title="Task6 TextOp - Isaac Ego" -- bash "$SELF" --run-block 3
    gnome-terminal --window --title="Task6 TextOp - Eval" -- bash "$SELF" --run-block 4
    ;;
  *) echo "Usage: $0" >&2; exit 2 ;;
esac
