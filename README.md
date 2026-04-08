---
title: CBIC Environment Server
emoji: 🧪
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - rl
  - spaces
  - customs
---

# CBIC Environment (OpenEnv)

OpenEnv-compatible RL environment for CBIC cargo inspection decisions.
The agent receives a Bill of Entry manifest, takes task-specific actions, and
gets scalar rewards from deterministic graders.

## RL Visibility (What Makes This RL)

| RL Component | In This Environment |
|---|---|
| Observation | Structured cargo manifest (importer, valuation, routing, compliance history) |
| Action | `detect_anomalies`, `assign_channel`, `draft_scn` |
| Reward | Numeric reward in `[0, 1]` at every step |
| Episode | `reset -> step... -> done` with 1/2/3-step tasks |
| Curriculum | Easy -> Medium -> Hard (`manifest-anomaly-detection`, `channel-assignment`, `show-cause-notice`) |
| Anti-gaming | SCN grader requires concrete numbers + valid legal sections + demand amount |

## Human Decision Simulation Features

The environment now includes structured officer-style reasoning signals in `details`
for each step, making trajectory rewards interpretable and more RL-trainable.

| Category | Feature |
|---|---|
| Drives score | Score anatomy (`base_reward`, trajectory bonus, penalties, final reward) |
| Drives score | Explicit expected action vs declared action alignment check |
| Flow | Step-1 pre-read narrative from manifest values |
| Flow | Step-2 deliberation summary before channel commitment |
| Context | Deterministic officer shift simulation (morning/afternoon/night) |
| Context | Port-aware queue pressure signal (low/moderate/high/critical) |
| Context | Escalation and 48-hour follow-up flags by channel severity |
| SCN task | Pre-draft SCN legal/figure planning object |
| RL shaping | Cross-step trajectory bonus linking step-1/2 quality to later rewards |

The inference runner prints scorer-compatible traces:

```text
[START] task=... env=india-customs-inspection model=...
[STEP] step=... action=... reward=... done=... error=...
[END] success=... steps=... score=... rewards=...
```

## Quick Start

```python
from cbic_rl import CbicEnv, CustomsAction

with CbicEnv(base_url="http://localhost:7860") as env:
    # Task 1: reset episode
    result = env.reset_http(task_name="manifest-anomaly-detection")
    print("BOE:", result.observation.manifest.boe_number)

    # Step with typed action
    result = env.step_http(
        CustomsAction(
            task="detect_anomalies",
            anomalies=["severe_undervaluation"],
        )
    )
    print(result.reward, result.done)
```

`CbicEnv` also implements OpenEnv `EnvClient` hooks (`_step_payload`,
`_parse_result`, `_parse_state`) for framework integrations.

## Building The Docker Image

```bash
docker build -t cbic-rl:latest -f Dockerfile .
```

## Environment Details

### Action

`StepRequest` controls interaction with the active task episode.

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | required | Action type (`detect_anomalies`, `assign_channel`, `draft_scn`) |
| `anomalies` | `list[str] \| None` | `None` | Predicted anomalies (step 1) |
| `channel` | `str \| None` | `None` | Channel decision `GREEN/ORANGE/RED` (step 2) |
| `notice_text` | `str \| None` | `None` | Show Cause Notice text (step 3) |
| `scn_text` | `str \| None` | `None` | Compatibility alias mapped to `notice_text` |

### Observation

`CargoManifest` is returned during reset and stored in state.

| Field | Type | Description |
|---|---|---|
| `boe_number` | `str` | Bill of Entry number |
| `port_of_entry` | `str` | Port where cargo arrived |
| `importer_name` | `str` | Importer legal name |
| `iec_code` | `str` | Importer Exporter Code |
| `iec_age_months` | `int` | IEC age in months |
| `country_of_origin` | `str` | Origin country |
| `country_of_export` | `str` | Exporting country |
| `commodity` | `str` | Goods description |
| `hs_code` | `str` | Tariff classification |
| `declared_weight_kg` | `float` | Declared cargo weight |
| `declared_value_usd` | `float` | Declared value |
| `market_value_usd` | `float \| None` | Reference market value |
| `previous_violations` | `int` | Prior violation count |
| `related_party` | `bool` | Buyer-seller related party flag |
| `related_party_disclosed` | `bool` | Whether related party was disclosed |
| `routing_countries` | `list[str]` | Shipment route path |
| `container_type` | `str` | Container class |
| `description` | `str` | Cargo detail string |

### State

`EnvironmentState` tracks active episode state.

| Field | Type | Description |
|---|---|---|
| `episode_id` | `str \| None` | Active episode id |
| `trace_id` | `str \| None` | Per-episode trace id for auditability |
| `task_name` | `str \| None` | Current task |
| `step` | `int` | Current step number |
| `max_steps` | `int` | Episode horizon |
| `done` | `bool` | Whether episode finished |
| `cumulative_reward` | `float` | Reward sum so far |
| `manifest` | `CargoManifest \| None` | Current case manifest |
| `decision_log` | `list[dict]` | Step-by-step audit trail (action, reward, rubric) |

## Task Modes

| Task | Steps | Difficulty | Goal |
|---|---|---|---|
| `manifest-anomaly-detection` | 1 | Easy | Find all anomalies with weighted precision/recall |
| `channel-assignment` | 2 | Medium | Detect anomalies, then assign examination channel |
| `show-cause-notice` | 3 | Hard | Complete full workflow and draft a legally grounded SCN |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode (`task_name`, optional `case_id`, optional `difficulty`) |
| `POST` | `/step` | Submit current action |
| `GET` | `/state` | Inspect current episode state |
| `GET` | `/health` | Liveness check |
| `GET` | `/tasks` | Returns task metadata |
| `GET` | `/explain-last` | Returns latest decision rationale payload |
| `WS` | `/ws` | OpenEnv-style WebSocket message flow (`reset`, `step`, `state`) |

## Reward Summary

| Task | Reward Design |
|---|---|
| Anomaly detection | Severity-weighted recall minus false-positive penalty |
| Channel assignment | Distance-aware channel scoring + consistency penalties |
| SCN drafting | Legal/quantitative rubric with anti-boilerplate controls |

Reward range is `[0.0, 1.0]` per step.

## Running The Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server on port 7860
python server.py

# Optional: run scorer-format inference loop
export HF_TOKEN=hf_xxx
export SERVER_URL=http://localhost:7860
python inference.py

# Optional: deterministic benchmark cases per task
export BENCHMARK_MODE=true
python inference.py
```

## Dependency Notes

- Core runtime uses FastAPI, Uvicorn, Pydantic, OpenAI, PyYAML, and HTTPX.
- `websockets` is intentionally included for `/ws` protocol support.
- `openenv-core` is intentionally included for OpenEnv-compatible typing/client integration.

## Docker Run

```bash
docker build -t cbic-rl:latest .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx --name cbic-rl-main cbic-rl:latest
```

## Project Structure

```text
cbic_rl/
├── Dockerfile
├── __init__.py
├── client.py
├── inference.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── server.py
└── environment/
    ├── __init__.py
    ├── cases.py
    ├── environment.py
    ├── graders.py
    └── models.py
```

