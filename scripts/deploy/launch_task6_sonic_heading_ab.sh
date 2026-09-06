#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop
BASE_LAUNCHER="${ROOT}/scripts/deploy/launch_eval_scq_simple_task6_sonic.sh"
AB_RUNNER="${ROOT}/scripts/deploy/run_task6_sonic_heading_ab.sh"
SELF="${ROOT}/scripts/deploy/launch_task6_sonic_heading_ab.sh"

wait_http() {
  local deadline=$((SECONDS + 600))
  until curl --noproxy '*' -fsS http://127.0.0.1:22095/config >/dev/null 2>&1; do
    (( SECONDS < deadline )) || {
      echo "Timed out waiting for GR00T SONIC on TCP 22095." >&2
      return 1
    }
    sleep 2
  done
  echo "[ready] GR00T SONIC TCP 22095"
}

wait_udp() {
  local deadline=$((SECONDS + 600))
  until ss -H -lun | awk '{print $4}' | grep -Eq '(^|:)23331$'; do
    (( SECONDS < deadline )) || {
      echo "Timed out waiting for Isaac Ego on UDP 23331." >&2
      return 1
    }
    sleep 2
  done
  echo "[ready] Isaac Ego UDP 23331"
}

case "${1:-}" in
  --run-ab)
    wait_http
    wait_udp
    exec "${AB_RUNNER}" both
    ;;
  "")
    [[ -n "${DISPLAY:-}" ]] || {
      echo "DISPLAY is not set; run this from the Ubuntu desktop session." >&2
      exit 1
    }
    command -v gnome-terminal >/dev/null || {
      echo "gnome-terminal is required." >&2
      exit 1
    }
    if ss -H -ltn | awk '{print $4}' | grep -Eq '(^|:)22095$'; then
      echo "TCP 22095 is already occupied." >&2
      exit 1
    fi
    if ss -H -lun | awk '{print $4}' | grep -Eq '(^|:)23331$'; then
      echo "UDP 23331 is already occupied." >&2
      exit 1
    fi
    rm -f /dev/shm/simple_task6_isaac_ego.frame
    gnome-terminal --window --title="Task6 Heading AB - GR00T" -- \
      bash "${BASE_LAUNCHER}" --run-block 1
    gnome-terminal --window --title="Task6 Heading AB - Isaac" -- \
      bash "${BASE_LAUNCHER}" --run-block 2
    gnome-terminal --window --title="Task6 Heading AB - Raw then Face Can" -- \
      bash "${SELF}" --run-ab
    ;;
  *)
    echo "Usage: $0" >&2
    exit 2
    ;;
esac
