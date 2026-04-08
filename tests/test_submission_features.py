import re

from fastapi.testclient import TestClient

from environment.cases import CASES
from environment.environment import CustomsEnvironment, TASK_ANOMALY, TASK_CHANNEL, TASK_SCN
from environment.graders import SCNGrader
from inference import (
    TASK_CONFIGS,
    format_start_line,
    format_step_line,
    format_end_line,
)
from server import app


def test_output_line_format_contract() -> None:
    start = format_start_line("manifest-anomaly-detection")
    step = format_step_line(1, "detect_anomalies", 0.75, False, "null")
    end = format_end_line(True, [0.75])

    start_re = re.compile(r"^\[START\] task=[a-z\-]+ env=india-customs-inspection model=.+$")
    step_re = re.compile(
        r"^\[STEP\] step=\d+ action=.* reward=\d+\.\d{2} done=(true|false) error=.*$"
    )
    end_re = re.compile(
        r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{2} rewards=\d+\.\d{2}(,\d+\.\d{2})*$"
    )

    assert start_re.match(start)
    assert step_re.match(step)
    assert end_re.match(end)


def test_reward_bounds_across_all_tasks() -> None:
    env = CustomsEnvironment(seed=7)

    # Task 1
    env.reset(task_name=TASK_ANOMALY)
    r1 = env.step({"task": "detect_anomalies", "anomalies": []}).reward
    assert 0.0 <= r1 <= 1.0

    # Task 2
    env.reset(task_name=TASK_CHANNEL)
    r2_1 = env.step({"task": "detect_anomalies", "anomalies": []}).reward
    r2_2 = env.step({"task": "assign_channel", "channel": "ORANGE"}).reward
    assert 0.0 <= r2_1 <= 1.0
    assert 0.0 <= r2_2 <= 1.0

    # Task 3
    env.reset(task_name=TASK_SCN)
    r3_1 = env.step(
        {
            "task": "extract_key_facts",
            "key_facts": {
                "declared_value_usd": 1000,
                "market_value_usd": 1200,
                "assessed_value_inr": 83000,
                "fx_rate_used": 83.0,
                "declared_weight_kg": 200,
                "country_of_origin": "China",
                "iec_age_months": 6,
            },
        }
    ).reward
    r3_2 = env.step({"task": "detect_anomalies", "anomalies": []}).reward
    r3_3 = env.step({"task": "rank_risk_severity", "ranked_anomalies": []}).reward
    r3_4 = env.step({"task": "assign_channel", "channel": "ORANGE"}).reward
    r3_5 = env.step({"task": "cite_legal_basis", "legal_sections": ["14", "114A"]}).reward
    r3_6 = env.step(
        {
            "task": "draft_scn",
            "notice_text": (
                "Under Section 14 and Section 114A, this notice is issued.\n\n"
                "Declared value is USD 1000 and weight is 200 kg with IEC age 6 months.\n\n"
                "A duty demand of INR 500000 and penalty proceedings are initiated."
            ),
        }
    ).reward
    r3_7 = env.step(
        {
            "task": "propose_enforcement",
            "enforcement_recommendation": "Recommend duty demand INR 500000 with penalty and confiscation proceedings.",
        }
    ).reward
    assert 0.0 <= r3_1 <= 1.0
    assert 0.0 <= r3_2 <= 1.0
    assert 0.0 <= r3_3 <= 1.0
    assert 0.0 <= r3_4 <= 1.0
    assert 0.0 <= r3_5 <= 1.0
    assert 0.0 <= r3_6 <= 1.0
    assert 0.0 <= r3_7 <= 1.0


def test_three_task_configs_present() -> None:
    names = {t["task_name"] for t in TASK_CONFIGS}
    assert len(TASK_CONFIGS) == 3
    assert names == {
        "manifest-anomaly-detection",
        "channel-assignment",
        "show-cause-notice",
    }


def test_scn_text_notice_text_mapping_in_http_api() -> None:
    client = TestClient(app)

    reset = client.post("/reset", json={"task_name": "show-cause-notice"})
    assert reset.status_code == 200

    step0 = client.post(
        "/step",
        json={
            "task": "extract_key_facts",
            "key_facts": {
                "declared_value_usd": 1200,
                "market_value_usd": 4800,
                "assessed_value_inr": 99600,
                "fx_rate_used": 83.0,
                "declared_weight_kg": 250,
                "country_of_origin": "China",
                "iec_age_months": 6,
            },
        },
    )
    assert step0.status_code == 200

    step1 = client.post("/step", json={"task": "detect_anomalies", "anomalies": []})
    assert step1.status_code == 200

    step2 = client.post("/step", json={"task": "rank_risk_severity", "ranked_anomalies": []})
    assert step2.status_code == 200

    step3 = client.post("/step", json={"task": "assign_channel", "channel": "ORANGE"})
    assert step3.status_code == 200

    step4 = client.post("/step", json={"task": "cite_legal_basis", "legal_sections": ["14", "114A"]})
    assert step4.status_code == 200

    step5 = client.post(
        "/step",
        json={
            "task": "draft_scn",
            "scn_text": (
                "Section 14 and Section 114A are invoked for misdeclaration.\n\n"
                "Declared value USD 1200 and market value USD 4800 indicate undervaluation.\n\n"
                "Duty demand INR 750000 with penalty and confiscation proceedings is proposed."
            ),
        },
    )

    assert step5.status_code == 200
    body = step5.json()
    assert "details" in body
    assert "legal_sections_score" in body["details"]


def test_scn_grader_detects_section_114a() -> None:
    grader = SCNGrader()
    case = CASES[0]

    notice_text = (
        f"This show cause notice invokes Section 14 and Section 114A for the goods in BOE {case.manifest.boe_number}.\n\n"
        f"Declared value is USD {int(case.manifest.declared_value_usd)} and weight is {int(case.manifest.declared_weight_kg)} kg, "
        f"while IEC age is {case.manifest.iec_age_months} months.\n\n"
        "A duty demand of INR 900000 with penalty and confiscation is proposed after adjudication."
    )

    reward, _, details = grader.grade(
        scn_text=notice_text,
        manifest=case.manifest,
        metadata=case.metadata,
        agent_anomalies=[a.value for a in case.metadata.true_anomalies],
    )

    assert 0.0 <= reward <= 1.0
    assert details["valid_sections_found"] >= 2
