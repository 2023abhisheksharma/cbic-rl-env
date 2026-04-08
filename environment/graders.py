"""
CBIC RL Environment Graders — Complete Rewrite
Three grader classes:
  - AnomalyDetectionGrader  : severity-weighted recall (Task 1)
  - ChannelAssignmentGrader : asymmetric tier-distance + cross-consistency (Task 2)
  - SCNGrader               : manifest-aware, 5-component (Task 3)

All 3 critical bug fixes from the audit report are applied:
  Fix #3: Number normalization (raw + comma-formatted, no ±10% float tolerance)
  Fix #5: Correct comment label — threshold ≥1.3 means HIGH OR CRITICAL
  Fix #6: agent_channel stored but documented as reserved (not used in SCNGrader)
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from environment.models import (
    AnomalyType, Channel, CargoManifest, CaseMetadata,
    ANOMALY_SEVERITY, ANOMALY_KEYWORDS,
    VALID_CUSTOMS_SECTIONS, VALID_CUSTOMS_SECTIONS_STR,
)


# ---------------------------------------------------------------------------
# 1. AnomalyDetectionGrader
# ---------------------------------------------------------------------------

class AnomalyDetectionGrader:
    """
    Grades anomaly detection using severity-weighted recall.
    Missing a CRITICAL anomaly (weight 1.5) costs more than missing BASE (1.0).
    """

    def grade(
        self,
        predicted_anomalies: List[str],
        metadata: CaseMetadata,
    ) -> tuple[float, str, Dict[str, Any]]:

        # --- normalize predictions ---
        valid_values = {a.value for a in AnomalyType}
        normalized_predictions: set[AnomalyType] = set()
        invalid_predictions: list[str] = []

        for pred in predicted_anomalies:
            if pred in valid_values:
                normalized_predictions.add(AnomalyType(pred))
            else:
                invalid_predictions.append(pred)

        true_set: set[AnomalyType] = set(metadata.true_anomalies)

        true_positives = normalized_predictions & true_set
        false_positives = normalized_predictions - true_set
        false_negatives = true_set - normalized_predictions

        # --- clean case (no true anomalies) ---
        if not true_set:
            fp_count = len(false_positives)
            reward = 1.0 if fp_count == 0 else max(0.0, 1.0 - 0.15 * fp_count)
            reward = round(reward, 4)
            feedback = (
                "Clean case correctly identified." if fp_count == 0
                else f"Clean case but {fp_count} false positive(s) flagged."
            )
            details = {
                "true_positives": [],
                "false_positives": [a.value for a in false_positives],
                "false_negatives": [],
                "invalid_predictions": invalid_predictions,
                "weighted_recall": 1.0,
                "total_true_anomalies": 0,
            }
            return reward, feedback, details

        # --- anomaly case ---
        total_weight = sum(ANOMALY_SEVERITY[a] for a in true_set)
        tp_weight = sum(ANOMALY_SEVERITY[a] for a in true_positives)
        fp_weight = sum(ANOMALY_SEVERITY.get(a, 1.0) for a in false_positives)

        weighted_recall = tp_weight / total_weight
        weighted_fp_penalty = 0.15 * fp_weight

        reward = max(0.0, min(1.0, weighted_recall - weighted_fp_penalty))
        reward = round(reward, 4)

        missed = [a.value for a in false_negatives]
        feedback_parts = []
        if true_positives:
            feedback_parts.append(
                f"Correctly flagged: {[a.value for a in true_positives]}."
            )
        if false_negatives:
            feedback_parts.append(f"Missed anomalies: {missed}.")
        if false_positives:
            feedback_parts.append(
                f"False positives: {[a.value for a in false_positives]}."
            )
        if invalid_predictions:
            feedback_parts.append(f"Invalid anomaly strings: {invalid_predictions}.")

        feedback = " ".join(feedback_parts) or "No anomalies detected."

        details = {
            "true_positives": [a.value for a in true_positives],
            "false_positives": [a.value for a in false_positives],
            "false_negatives": [a.value for a in false_negatives],
            "invalid_predictions": invalid_predictions,
            "weighted_recall": round(weighted_recall, 4),
            "total_true_anomalies": len(true_set),
        }
        return reward, feedback, details


# ---------------------------------------------------------------------------
# 2. ChannelAssignmentGrader
# ---------------------------------------------------------------------------

_CHANNEL_ORDER = [Channel.GREEN, Channel.ORANGE, Channel.RED]

# Base scoring matrix: (assigned_index, correct_index) → base_reward
_BASE_SCORE_MATRIX: dict[tuple[int, int], float] = {
    (0, 0): 1.0,   # GREEN when GREEN — exact
    (1, 1): 1.0,   # ORANGE when ORANGE — exact
    (2, 2): 1.0,   # RED when RED — exact
    # one tier too strict (over-inspecting)
    (1, 0): 0.6,   # ORANGE when GREEN correct
    (2, 1): 0.6,   # RED when ORANGE correct
    # one tier too lenient
    (0, 1): 0.5,   # GREEN when ORANGE correct (clean case, mild danger)
    # The next two depend on whether anomalies exist — handled in grade()
    (1, 2): 0.25,  # ORANGE when RED correct (anomalies present — dangerous)
    (0, 2): 0.1,   # GREEN when RED correct — critically dangerous
    # two tiers too strict
    (2, 0): 0.25,  # RED when GREEN — significant resource waste
}


class ChannelAssignmentGrader:
    """
    Grades channel assignment (GREEN/ORANGE/RED).
    Includes cross-consistency check with agent's own anomaly list.
    Fix #5: threshold ≥1.3 covers HIGH AND CRITICAL severity anomalies.
    """

    def grade(
        self,
        assigned_channel: str,
        metadata: CaseMetadata,
        agent_anomalies: Optional[List[str]] = None,
    ) -> tuple[float, str, Dict[str, Any]]:

        # --- validate channel string ---
        try:
            assigned = Channel(assigned_channel.upper())
        except (ValueError, AttributeError):
            return 0.0, f"Invalid channel: '{assigned_channel}'.", {
                "assigned": assigned_channel,
                "correct": metadata.correct_channel.value,
                "base_score": 0.0,
                "consistency_penalty": 0.0,
            }

        correct = metadata.correct_channel
        assigned_idx = _CHANNEL_ORDER.index(assigned)
        correct_idx = _CHANNEL_ORDER.index(correct)

        # --- base score ---
        base_score = _BASE_SCORE_MATRIX.get((assigned_idx, correct_idx), 0.0)

        # Special case: GREEN when ORANGE correct but case is NOT anomalous
        # (already handled by matrix — GREEN when ORANGE is 0.5 regardless)

        reward = base_score
        feedback = f"Assigned {assigned.value}, correct was {correct.value}."

        if assigned == correct:
            feedback = f"Correct channel {assigned.value} assigned."

        # --- cross-consistency check ---
        # Fix #5 comment: ≥1.3 means HIGH OR CRITICAL (not only CRITICAL)
        consistency_penalty = 0.0
        if agent_anomalies:
            high_or_critical = [
                a for a in agent_anomalies
                if ANOMALY_SEVERITY.get(AnomalyType(a), 1.0) >= 1.3
                if a in {at.value for at in AnomalyType}
            ]
            if high_or_critical and assigned in (Channel.GREEN, Channel.ORANGE):
                consistency_penalty = 0.20
                reward = max(0.0, reward - consistency_penalty)
                feedback += (
                    f" Consistency penalty: HIGH/CRITICAL anomalies flagged "
                    f"({high_or_critical}) but lenient channel assigned."
                )

        reward = round(reward, 4)

        details = {
            "assigned": assigned.value,
            "correct": correct.value,
            "base_score": base_score,
            "consistency_penalty": consistency_penalty,
        }
        return reward, feedback, details


# ---------------------------------------------------------------------------
# 3. SCNGrader
# ---------------------------------------------------------------------------

def _normalize_number(val: float) -> set[str]:
    """Return raw int string and comma-formatted string for matching.
    Fix #3: no ±10% float tolerance — instead handle both 1200 and 1,200 forms.
    """
    n = int(val)
    return {str(n), f"{n:,}"}


# Regex for legal section citations
_SECTION_RE = re.compile(r'[Ss]ection\s+(\d{1,3}[A-Za-z]?)')

# Regex for demand amounts
_DEMAND_RE = re.compile(
    r'(?:Rs\.?|INR|USD)\s*[\d,]+(?:\.\d+)?\s*(?:crore|lakh|thousand)?'
    r'|\b\d{5,}\b'  # any 5+ digit number near enforcement keywords
)

_ENFORCEMENT_KEYWORDS = [
    "show cause", "demand", "penalty", "confiscat", "seize", "adjudicate",
    "duty evad", "fine", "forfeiture",
]


class SCNGrader:
    """
    Grades Show Cause Notice drafts.
    5-component scoring weighted to prevent boilerplate gaming.
    agent_channel stored in episode state but not used here (Fix #6: reserved).
    """

    def grade(
        self,
        scn_text: str,
        manifest: CargoManifest,
        metadata: CaseMetadata,
        agent_anomalies: Optional[List[str]] = None,
    ) -> tuple[float, str, Dict[str, Any]]:

        if not scn_text or not scn_text.strip():
            return 0.0, "Empty SCN submitted.", {
                "manifest_facts_score": 0.0,
                "legal_sections_score": 0.0,
                "anomaly_coverage_score": 0.0,
                "enforcement_score": 0.0,
                "length_structure_score": 0.0,
            }

        text = scn_text

        # ---------------------------------------------------------------
        # Component 1 (25%): Manifest-specific facts cited
        # Fix #3: normalize to both raw and comma-formatted forms
        # ---------------------------------------------------------------
        manifest_numbers: set[str] = set()

        for field in ["declared_value_usd", "market_value_usd", "declared_weight_kg"]:
            val = getattr(manifest, field, None)
            if val:
                manifest_numbers.update(_normalize_number(val))

        if manifest.iec_age_months:
            manifest_numbers.add(str(manifest.iec_age_months))

        cited_count = sum(1 for n in manifest_numbers if n in text)
        manifest_facts_score = min(cited_count / 2, 1.0)

        # ---------------------------------------------------------------
        # Component 2 (25%): Legal section citations
        # Fix #4: Section 18 is in VALID_CUSTOMS_SECTIONS (already in models.py)
        # ---------------------------------------------------------------
        found_sections = _SECTION_RE.findall(text)
        valid_hits = 0
        for sec in found_sections:
            try:
                if int(re.sub(r'[A-Za-z]', '', sec)) in VALID_CUSTOMS_SECTIONS:
                    valid_hits += 1
                    continue
            except ValueError:
                pass
            if sec in VALID_CUSTOMS_SECTIONS_STR:
                valid_hits += 1

        legal_sections_score = min(valid_hits / 2, 1.0)

        # ---------------------------------------------------------------
        # Component 3 (20%): Anomaly coverage (agent's own findings)
        # ---------------------------------------------------------------
        anomaly_coverage_score = 0.0
        if agent_anomalies:
            covered = 0
            for anomaly_str in agent_anomalies:
                try:
                    atype = AnomalyType(anomaly_str)
                    keywords = ANOMALY_KEYWORDS.get(atype, [])
                    if any(kw.lower() in text.lower() for kw in keywords):
                        covered += 1
                except ValueError:
                    pass
            if agent_anomalies:
                anomaly_coverage_score = covered / len(agent_anomalies)
        else:
            anomaly_coverage_score = 0.5  # neutral if no anomaly list

        # ---------------------------------------------------------------
        # Component 4 (20%): Enforcement action + demand amount
        # ---------------------------------------------------------------
        text_lower = text.lower()
        has_enforcement_keywords = any(kw in text_lower for kw in _ENFORCEMENT_KEYWORDS)
        demand_matches = _DEMAND_RE.findall(text)
        has_demand_figure = len(demand_matches) > 0

        if has_enforcement_keywords and has_demand_figure:
            enforcement_score = 1.0
        elif has_enforcement_keywords:
            enforcement_score = 0.5
        else:
            enforcement_score = 0.0

        # ---------------------------------------------------------------
        # Component 5 (10%): Length and structure
        # ---------------------------------------------------------------
        word_count = len(text.split())
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        length_ok = word_count >= 150
        structure_ok = paragraph_count >= 3
        length_structure_score = 0.5 * length_ok + 0.5 * structure_ok

        # ---------------------------------------------------------------
        # Final weighted reward
        # ---------------------------------------------------------------
        reward = (
            0.25 * manifest_facts_score
            + 0.25 * legal_sections_score
            + 0.20 * anomaly_coverage_score
            + 0.20 * enforcement_score
            + 0.10 * length_structure_score
        )
        reward = round(min(1.0, max(0.0, reward)), 4)

        feedback_parts = [
            f"Manifest facts cited: {cited_count} (score {manifest_facts_score:.2f}).",
            f"Valid legal sections: {valid_hits} (score {legal_sections_score:.2f}).",
            f"Anomaly coverage: {anomaly_coverage_score:.2f}.",
            f"Enforcement score: {enforcement_score:.2f}.",
            f"Length/structure: {word_count} words, {paragraph_count} paragraphs "
            f"(score {length_structure_score:.2f}).",
        ]
        feedback = " | ".join(feedback_parts)

        details = {
            "manifest_facts_score": manifest_facts_score,
            "legal_sections_score": legal_sections_score,
            "anomaly_coverage_score": anomaly_coverage_score,
            "enforcement_score": enforcement_score,
            "length_structure_score": length_structure_score,
            "word_count": word_count,
            "paragraph_count": paragraph_count,
            "valid_sections_found": valid_hits,
            "manifest_numbers_cited": cited_count,
        }
        return reward, feedback, details

