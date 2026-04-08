"""
CBIC RL Environment — Inference Script
Runs 3 independent task episodes sequentially using OpenAI-compatible client.
Prints exact [START]/[STEP]/[END] format required by the organizer scorer.

Fixes applied:
  Fix #2:  scn_text → notice_text field name translation
  Fix #4:  Section 18 added to Step 3 valid sections in system prompt
  Fix #8:  SERVER_URL has hardcoded fallback default
"""

import os
import json
import re

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variable setup
# Fix #8: SERVER_URL always has a fallback default
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")  # Fix #8

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

STEP1_SYSTEM = """You are a CBIC customs risk intelligence officer. Analyze the cargo
manifest and return ONLY valid JSON with no other text. The JSON must have exactly
one key: 'anomalies' containing a list of anomaly type strings.

Valid anomaly types: weight_volume_mismatch, severe_undervaluation,
suspicious_routing, new_iec_high_value, high_risk_origin,
undisclosed_related_party, repeat_violator, hs_code_risk

Return [] if no anomalies detected.
Example: {"anomalies": ["severe_undervaluation", "high_risk_origin"]}"""

STEP2_SYSTEM = """You are a CBIC customs officer assigning examination channels.
Based on the manifest and anomalies already detected, assign the correct channel.
Return ONLY valid JSON: {"channel": "GREEN" | "ORANGE" | "RED"}

GREEN = auto-clear (no anomalies detected).
ORANGE = document check only (low-medium risk anomalies).
RED = mandatory physical inspection (serious anomalies found)."""

# Fix #4: Section 18 added to the valid sections list
STEP3_SYSTEM = """You are a senior CBIC customs officer drafting a Show Cause Notice.
You MUST include in the SCN:
1. At least 2 specific numbers from the manifest
   (declared value, market value, weight, IEC age in months)
2. At least 2 real Customs Act 1962 section numbers
   (valid: 14, 18, 46, 47, 111, 112, 113, 114, 114A, 127)
3. A specific duty/penalty demand amount in INR or USD
4. At least 3 paragraphs. Minimum 150 words.

Return ONLY valid JSON: {"scn_text": "<full SCN text here>"}"""

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIGS = [
    {
        "task_name": "manifest-anomaly-detection",
        "steps": 1,
        "actions": ["detect_anomalies"],
    },
    {
        "task_name": "channel-assignment",
        "steps": 2,
        "actions": ["detect_anomalies", "assign_channel"],
    },
    {
        "task_name": "show-cause-notice",
        "steps": 3,
        "actions": ["detect_anomalies", "assign_channel", "draft_scn"],
    },
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def post_reset(http: httpx.Client, task_name: str) -> dict:
    resp = http.post(f"{SERVER_URL}/reset", json={"task_name": task_name})
    resp.raise_for_status()
    return resp.json()


def post_step(http: httpx.Client, payload: dict) -> dict:
    resp = http.post(f"{SERVER_URL}/step", json=payload)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call the LLM and return raw content string."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1000,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def parse_json_safe(content: str, fallback: dict) -> dict:
    """Try to parse JSON from LLM output with fallback."""
    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Regex search for JSON object
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return fallback


# ---------------------------------------------------------------------------
# Build user prompts from manifest
# ---------------------------------------------------------------------------

def manifest_to_text(manifest: dict) -> str:
    """Convert manifest dict to a readable text block."""
    lines = [
        f"BOE Number: {manifest.get('boe_number')}",
        f"Port of Entry: {manifest.get('port_of_entry')}",
        f"Importer: {manifest.get('importer_name')}",
        f"IEC Code: {manifest.get('iec_code')} (age: {manifest.get('iec_age_months')} months)",
        f"Country of Origin: {manifest.get('country_of_origin')}",
        f"Country of Export: {manifest.get('country_of_export')}",
        f"Routing Countries: {', '.join(manifest.get('routing_countries', []))}",
        f"Commodity: {manifest.get('commodity')}",
        f"HS Code: {manifest.get('hs_code')}",
        f"Declared Weight: {manifest.get('declared_weight_kg')} kg",
        f"Declared Value: USD {manifest.get('declared_value_usd')}",
        f"Market Value: USD {manifest.get('market_value_usd', 'N/A')}",
        f"Previous Violations: {manifest.get('previous_violations')}",
        f"Related Party: {manifest.get('related_party')} "
        f"(Disclosed: {manifest.get('related_party_disclosed')})",
        f"Container Type: {manifest.get('container_type')}",
        f"Description: {manifest.get('description')}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_task(task_config: dict) -> None:
    task_name = task_config["task_name"]
    actions = task_config["actions"]

    # Print [START] line
    print(f"[START] task={task_name} env=india-customs-inspection model={MODEL_NAME}")

    rewards: list[float] = []
    success = True

    try:
        with httpx.Client(timeout=90.0) as http:
            # Reset environment
            reset_data = post_reset(http, task_name)
            manifest = reset_data.get("manifest", {})
            manifest_text = manifest_to_text(manifest)

            # State carried between steps
            detected_anomalies: list[str] = []
            assigned_channel: str = ""

            for step_num, action_name in enumerate(actions, start=1):
                error_str = "null"
                reward = 0.0
                done_str = "false"

                try:
                    if action_name == "detect_anomalies":
                        user_prompt = (
                            f"Analyze this cargo manifest and detect all anomalies:\n\n"
                            f"{manifest_text}"
                        )
                        content = call_llm(STEP1_SYSTEM, user_prompt)
                        parsed = parse_json_safe(content, {"anomalies": []})
                        detected_anomalies = parsed.get("anomalies", [])

                        payload = {
                            "task": "detect_anomalies",
                            "anomalies": detected_anomalies,
                        }

                    elif action_name == "assign_channel":
                        user_prompt = (
                            f"Manifest:\n{manifest_text}\n\n"
                            f"Anomalies detected in step 1: {detected_anomalies}\n\n"
                            f"Assign the correct examination channel."
                        )
                        content = call_llm(STEP2_SYSTEM, user_prompt)
                        parsed = parse_json_safe(content, {"channel": "ORANGE"})
                        assigned_channel = parsed.get("channel", "ORANGE")

                        payload = {
                            "task": "assign_channel",
                            "channel": assigned_channel,
                        }

                    elif action_name == "draft_scn":
                        user_prompt = (
                            f"Manifest:\n{manifest_text}\n\n"
                            f"Anomalies detected: {detected_anomalies}\n"
                            f"Channel assigned: {assigned_channel}\n\n"
                            f"Draft a detailed Show Cause Notice citing specific manifest "
                            f"figures and Customs Act sections."
                        )
                        content = call_llm(STEP3_SYSTEM, user_prompt)
                        parsed = parse_json_safe(content, {"scn_text": ""})

                        # Fix #2: translate scn_text → notice_text for the server
                        scn_text = parsed.get("scn_text", "")
                        payload = {
                            "task": "draft_scn",
                            "notice_text": scn_text,  # Fix #2: correct field name
                        }

                    # Submit step to environment
                    step_result = post_step(http, payload)
                    reward = step_result.get("reward", 0.0)
                    done = step_result.get("done", False)
                    done_str = "true" if done else "false"

                    last_action_error = step_result.get("last_action_error")
                    if not last_action_error:
                        last_action_error = (step_result.get("details") or {}).get("last_action_error")
                    if last_action_error:
                        error_str = str(last_action_error).replace("\n", " ")

                except Exception as e:
                    error_str = str(e).replace("\n", " ")
                    done_str = "true" if step_num == len(actions) else "false"
                    success = False

                rewards.append(reward)
                print(
                    f"[STEP] step={step_num} action={action_name} "
                    f"reward={reward:.2f} done={done_str} error={error_str}"
                )

    except Exception:
        # Emit [END] even on exception
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        score = (sum(rewards) / len(rewards)) if rewards else 0.0
        print(
            f"[END] success=false steps={len(rewards)} score={score:.2f} rewards={rewards_str}"
        )
        return

    # Print [END] line
    steps_taken = len(rewards)
    score = (sum(rewards) / steps_taken) if steps_taken else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps_taken} score={score:.2f} rewards={rewards_str}"
    )


def main():
    for task_config in TASK_CONFIGS:
        run_task(task_config)


if __name__ == "__main__":
    main()

