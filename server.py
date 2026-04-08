"""
CBIC RL Environment — FastAPI HTTP Server
Endpoints: POST /reset, POST /step, GET /state, GET /health, GET /tasks, GET /explain-last
Port: 7860

Fixes applied:
  Fix #1:  task_name is Optional in ResetRequest (HEALTHCHECK works with empty body)
  Fix #10: HEALTHCHECK skip-if-active guard via ?healthcheck=true query param
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from environment import (
    CustomsEnvironment,
    ResetRequest,
    StepRequest,
    ResetResponse,
    StepResponse,
    EnvironmentState,
)
from environment.environment import VALID_TASKS, TASK_STEPS, EXPECTED_ACTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENV_SEED = int(os.getenv("ENV_SEED", "42"))

# Global environment instance
env = CustomsEnvironment(seed=ENV_SEED)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"CBIC RL Environment starting. Seed={ENV_SEED}, Cases loaded={60}")
    yield
    logger.info("CBIC RL Environment shutting down.")


app = FastAPI(
    title="CBIC Customs Inspection RL Environment",
    description="RL environment for India Customs cargo inspection decisions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
async def reset(
    req: ResetRequest,
    healthcheck: bool = Query(default=False),
):
    """
    Start a new episode.
    Fix #1: task_name defaults to 'manifest-anomaly-detection' if not provided.
    Fix #10: If healthcheck=true and an episode is active, skip reset to protect it.
    """
    # Fix #10: HEALTHCHECK guard
    if healthcheck and env.is_episode_active():
        return {
            "status": "ok",
            "skipped": True,
            "reason": "episode active",
            # Return dummy response fields to satisfy response model
            "episode_id": "HEALTHCHECK",
            "task_name": req.task_name,
            "manifest": env.get_state().manifest,
            "step": 0,
            "max_steps": 1,
        }

    try:
        response = env.reset(task_name=req.task_name, case_id=req.case_id, difficulty=req.difficulty)
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    """Process one agent action in the current episode."""
    notice_text = req.notice_text or req.scn_text or ""
    action = {
        "task": req.task,
        "anomalies": req.anomalies or [],
        "channel": req.channel or "",
        "notice_text": notice_text,
    }
    try:
        response = env.step(action)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

@app.get("/state", response_model=EnvironmentState)
async def state():
    """Return current environment state."""
    return env.get_state()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def tasks():
    """List all available tasks with metadata."""
    return {
        "rubric_keys": [
            "base_reward",
            "task_alignment",
            "action_validity",
            "trajectory_consistency",
            "final_reward",
        ],
        "tasks": [
            {
                "name": "manifest-anomaly-detection",
                "difficulty": "easy",
                "steps": 1,
                "expected_actions": EXPECTED_ACTIONS["manifest-anomaly-detection"],
                "reward_components": ["anomaly_detection", "task_alignment", "action_validity"],
                "failure_modes": ["invalid_anomaly_label", "task_mismatch", "false_positive_burst"],
                "description": (
                    "Detect all anomaly types present in the manifest. "
                    "Severity-weighted recall scoring."
                ),
            },
            {
                "name": "channel-assignment",
                "difficulty": "medium",
                "steps": 2,
                "expected_actions": EXPECTED_ACTIONS["channel-assignment"],
                "reward_components": ["anomaly_detection", "channel_correctness", "consistency"],
                "failure_modes": ["lenient_channel_for_high_risk", "invalid_channel", "cross_step_inconsistency"],
                "description": (
                    "Detect anomalies then assign correct CBIC examination channel. "
                    "Cross-consistency check applies."
                ),
            },
            {
                "name": "show-cause-notice",
                "difficulty": "hard",
                "steps": 3,
                "expected_actions": EXPECTED_ACTIONS["show-cause-notice"],
                "reward_components": ["manifest_facts", "legal_sections", "anomaly_coverage", "enforcement", "structure"],
                "failure_modes": ["template_scn", "missing_legal_basis", "red_channel_without_linked_enforcement"],
                "description": (
                    "Full pipeline ending in SCN draft citing actual manifest "
                    "figures and legal Customs Act sections."
                ),
            },
        ]
    }


@app.get("/explain-last")
async def explain_last():
    """Return the latest decision rationale for judge explainability checks."""
    return env.get_last_explain()


# ---------------------------------------------------------------------------
# WS /ws (OpenEnv-compatible message flow)
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """
    Minimal WebSocket endpoint compatible with EnvClient message types.

    Expected inbound messages:
      {"type": "reset", "data": {...}}
      {"type": "step", "data": {...}}
      {"type": "state"}

    Returns envelope messages with top-level `data`.
    """
    await websocket.accept()
    ws_env = CustomsEnvironment(seed=ENV_SEED)

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = (message or {}).get("type", "")
            data = (message or {}).get("data", {}) or {}

            if msg_type == "reset":
                task_name = data.get("task_name", "manifest-anomaly-detection")
                case_id = data.get("case_id")
                difficulty = data.get("difficulty")
                reset_result = ws_env.reset(task_name=task_name, case_id=case_id, difficulty=difficulty)
                payload = reset_result.model_dump()
                response = {
                    "type": "reset_result",
                    "data": {
                        "observation": {
                            "task_name": payload.get("task_name"),
                            "step": payload.get("step", 0),
                            "max_steps": payload.get("max_steps", 0),
                            "manifest": payload.get("manifest"),
                            "feedback": "",
                            "details": {},
                            "cumulative_reward": 0.0,
                            "done": False,
                        },
                        "reward": 0.0,
                        "done": False,
                    },
                }
                await websocket.send_json(response)

            elif msg_type == "step":
                step_data = dict(data)
                if "notice_text" not in step_data and "scn_text" in step_data:
                    step_data["notice_text"] = step_data.get("scn_text", "")
                step_result = ws_env.step(step_data)
                state = ws_env.get_state()
                payload = step_result.model_dump()
                response = {
                    "type": "step_result",
                    "data": {
                        "observation": {
                            "task_name": state.task_name,
                            "step": payload.get("step", 0),
                            "max_steps": state.max_steps,
                            "manifest": state.manifest.model_dump() if state.manifest else None,
                            "feedback": payload.get("feedback", ""),
                            "details": payload.get("details", {}),
                            "cumulative_reward": payload.get("cumulative_reward", 0.0),
                            "done": payload.get("done", False),
                        },
                        "reward": payload.get("reward", 0.0),
                        "done": payload.get("done", False),
                    },
                }
                await websocket.send_json(response)

            elif msg_type == "state":
                state = ws_env.get_state()
                response = {
                    "type": "state_result",
                    "data": {
                        "episode_id": state.episode_id,
                        "step_count": state.step,
                        "task_name": state.task_name,
                        "max_steps": state.max_steps,
                        "done": state.done,
                        "cumulative_reward": state.cumulative_reward,
                    },
                }
                await websocket.send_json(response)

            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "message": f"Unsupported message type: {msg_type}"
                        },
                    }
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console entrypoint for running the API server."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

