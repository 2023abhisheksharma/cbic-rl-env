#!/usr/bin/env bash
# Local pre-submit validator for CBIC OpenEnv hackathon submission.
# Usage:
#   scripts/validate-submission.sh [ping_url] [repo_dir]
# Examples:
#   scripts/validate-submission.sh
#   scripts/validate-submission.sh https://your-space.hf.space
#   SKIP_DOCKER=1 scripts/validate-submission.sh

set -u

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-600}"
SKIP_DOCKER="${SKIP_DOCKER:-0}"

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
SERVER_PID=""
SERVER_STARTED=0

pass() {
  printf "%bPASS%b %s\n" "$GREEN" "$NC" "$1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  printf "%bFAIL%b %s\n" "$RED" "$NC" "$1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn() {
  printf "%bWARN%b %s\n" "$YELLOW" "$NC" "$1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

run_with_timeout() {
  local secs="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@"
  fi
}

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if ! cd "$REPO_DIR"; then
  printf "%bFAIL%b Cannot cd to repo_dir: %s\n" "$RED" "$NC" "$REPO_DIR"
  exit 2
fi

printf "%b== OpenEnv Local Pre-Submission Validation ==%b\n" "$BOLD" "$NC"
printf "Repo: %s\n" "$PWD"

# -----------------------------------------------------------------------------
# 1) Static file checks
# -----------------------------------------------------------------------------
[ -f "inference.py" ] && pass "inference.py exists at repo root" || fail "inference.py missing at repo root"
[ -f "openenv.yaml" ] && pass "openenv.yaml exists" || fail "openenv.yaml missing"
[ -f "Dockerfile" ] && pass "Dockerfile exists" || fail "Dockerfile missing"
[ -f "README.md" ] && pass "README.md exists" || fail "README.md missing"

# -----------------------------------------------------------------------------
# 2) Inference implementation checks
# -----------------------------------------------------------------------------
if grep -Eq 'from openai import OpenAI|import openai' inference.py; then
  pass "inference.py uses OpenAI client import"
else
  fail "inference.py does not import OpenAI client"
fi

if grep -Eq 'API_BASE_URL\s*=\s*os\.getenv\(\s*["\x27]API_BASE_URL["\x27]\s*,' inference.py; then
  pass "API_BASE_URL has default"
else
  fail "API_BASE_URL default missing in inference.py"
fi

if grep -Eq 'MODEL_NAME\s*=\s*os\.getenv\(\s*["\x27]MODEL_NAME["\x27]\s*,' inference.py; then
  pass "MODEL_NAME has default"
else
  fail "MODEL_NAME default missing in inference.py"
fi

if grep -Eq 'HF_TOKEN\s*=\s*os\.getenv\(\s*["\x27]HF_TOKEN["\x27]\s*\)' inference.py; then
  pass "HF_TOKEN read without default"
else
  fail "HF_TOKEN must be read without default"
fi

if grep -Eq 'HF_TOKEN environment variable is required' inference.py; then
  pass "HF_TOKEN mandatory guard present"
else
  fail "HF_TOKEN mandatory guard missing"
fi

# -----------------------------------------------------------------------------
# 3) Start server (if not running) and endpoint checks
# -----------------------------------------------------------------------------
if curl -fsS http://localhost:7860/health >/dev/null 2>&1; then
  pass "Server already reachable on :7860"
else
  if [ -x "venv/bin/python" ]; then
    venv/bin/python server.py >/tmp/cbic_validate_server.log 2>&1 &
    SERVER_PID=$!
    SERVER_STARTED=1
  else
    python server.py >/tmp/cbic_validate_server.log 2>&1 &
    SERVER_PID=$!
    SERVER_STARTED=1
  fi

  SERVER_UP=0
  for _ in $(seq 1 20); do
    if curl -fsS http://localhost:7860/health >/dev/null 2>&1; then
      SERVER_UP=1
      break
    fi
    sleep 1
  done

  if [ "$SERVER_UP" = "1" ]; then
    pass "Server started successfully on :7860"
  else
    fail "Server failed to start on :7860"
  fi
fi

if curl -fsS http://localhost:7860/health >/dev/null 2>&1; then
  pass "GET /health responds 200"
else
  fail "GET /health failed"
fi

RESET_JSON="$(curl -sS -X POST http://localhost:7860/reset -H 'Content-Type: application/json' -d '{"task_name":"manifest-anomaly-detection"}' 2>/dev/null || true)"
if [ -n "$RESET_JSON" ] && printf "%s" "$RESET_JSON" | grep -q '"episode_id"'; then
  pass "POST /reset responds with episode_id"
else
  fail "POST /reset failed or malformed response"
fi

STEP_JSON="$(curl -sS -X POST http://localhost:7860/step -H 'Content-Type: application/json' -d '{"task":"detect_anomalies","anomalies":["severe_undervaluation"]}' 2>/dev/null || true)"
if [ -n "$STEP_JSON" ] && printf "%s" "$STEP_JSON" | grep -q '"reward"'; then
  pass "POST /step returns reward"
else
  fail "POST /step failed or missing reward"
fi

if [ -n "$STEP_JSON" ]; then
  if python - "$STEP_JSON" <<'PY' >/dev/null 2>&1
import json, sys
obj = json.loads(sys.argv[1])
r = float(obj.get('reward', -1))
assert 0.0 <= r <= 1.0
PY
  then
    pass "Reward from /step is in [0,1]"
  else
    fail "Reward from /step is outside [0,1]"
  fi
fi

TASKS_JSON="$(curl -sS http://localhost:7860/tasks 2>/dev/null || true)"
if [ -n "$TASKS_JSON" ]; then
  if python - "$TASKS_JSON" <<'PY' >/dev/null 2>&1
import json, sys
obj = json.loads(sys.argv[1])
assert isinstance(obj.get('tasks'), list)
assert len(obj['tasks']) >= 3
PY
  then
    pass "At least 3 tasks exposed via /tasks"
  else
    fail "Task enumeration invalid or fewer than 3 tasks"
  fi
else
  fail "GET /tasks failed"
fi

# WS compatibility check
if [ -x "venv/bin/python" ]; then
  PYBIN="venv/bin/python"
else
  PYBIN="python"
fi

if "$PYBIN" - <<'PY' >/dev/null 2>&1
import asyncio, json
import websockets

async def main():
    async with websockets.connect('ws://localhost:7860/ws') as ws:
        await ws.send(json.dumps({'type':'reset','data':{'task_name':'manifest-anomaly-detection'}}))
        r1 = json.loads(await ws.recv())
        await ws.send(json.dumps({'type':'state'}))
        r2 = json.loads(await ws.recv())
        assert r1.get('type') == 'reset_result'
        assert r2.get('type') == 'state_result'

asyncio.run(main())
PY
then
  pass "WebSocket /ws responds to reset/state protocol"
else
  warn "WebSocket /ws check failed"
fi

# -----------------------------------------------------------------------------
# 4) Inference stdout format simulation (no external API call)
# -----------------------------------------------------------------------------
if "$PYBIN" - <<'PY' >/tmp/cbic_inference_format_check.txt 2>&1
import io, os, re, contextlib

os.environ['HF_TOKEN'] = 'dummy-token'
os.environ['API_BASE_URL'] = 'https://router.huggingface.co/v1'
os.environ['MODEL_NAME'] = 'Qwen/Qwen2.5-72B-Instruct'

import inference


def fake_reset(http, task_name, case_id=None):
    return {'manifest': {'boe_number':'X','port_of_entry':'JNPT','importer_name':'A','iec_code':'IEC','iec_age_months':6,'country_of_origin':'IN','country_of_export':'IN','routing_countries':['IN'],'commodity':'C','hs_code':'1','declared_weight_kg':1,'declared_value_usd':1,'market_value_usd':2,'previous_violations':0,'related_party':False,'related_party_disclosed':True,'container_type':'20GP','description':'d'}}

def fake_step(http, payload):
    return {'reward': 0.5, 'done': True, 'details': {}}

def fake_llm(system_prompt, user_prompt):
    return '{"anomalies": ["severe_undervaluation"]}'

inference.post_reset = fake_reset
inference.post_step = fake_step
inference.call_llm = fake_llm

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    inference.run_task({'task_name':'manifest-anomaly-detection','actions':['detect_anomalies']})

out = [line.strip() for line in buf.getvalue().splitlines() if line.strip()]
assert len(out) == 3
assert re.match(r'^\[START\] task=.* env=.* model=.*$', out[0])
assert re.match(r'^\[STEP\] step=\d+ action=.* reward=\d+\.\d{2} done=(true|false) error=.*$', out[1])
assert re.match(r'^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=\d+\.\d{2}(,\d+\.\d{2})*$', out[2])
PY
then
  pass "inference.py emits required [START]/[STEP]/[END] format"
else
  fail "inference.py output format check failed (see /tmp/cbic_inference_format_check.txt)"
fi

# -----------------------------------------------------------------------------
# 5) Optional HF Space ping
# -----------------------------------------------------------------------------
if [ -n "$PING_URL" ]; then
  if curl -fsS "$PING_URL" >/dev/null 2>&1; then
    pass "HF Space ping URL reachable: $PING_URL"
  else
    fail "HF Space ping URL not reachable: $PING_URL"
  fi
else
  warn "No ping_url provided; skipping live HF Space check"
fi

# -----------------------------------------------------------------------------
# 6) Docker build check
# -----------------------------------------------------------------------------
if [ "$SKIP_DOCKER" = "1" ]; then
  warn "SKIP_DOCKER=1 set; skipping docker build check"
else
  if ! command -v docker >/dev/null 2>&1; then
    fail "Docker not found in PATH"
  else
    if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t cbic-rl-validate:local -f Dockerfile . >/tmp/cbic_docker_build.log 2>&1; then
      pass "docker build succeeded"
    else
      fail "docker build failed or timed out (${DOCKER_BUILD_TIMEOUT}s). See /tmp/cbic_docker_build.log"
    fi
  fi
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf "\n%bSummary%b: PASS=%d FAIL=%d WARN=%d\n" "$BOLD" "$NC" "$PASS_COUNT" "$FAIL_COUNT" "$WARN_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
  printf "%bResult: FAIL%b\n" "$RED" "$NC"
  exit 1
fi

printf "%bResult: PASS%b\n" "$GREEN" "$NC"
exit 0
