"""
Pydantic data models for the CBIC RL Environment.
All models use Pydantic v2 syntax.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Support OpenEnv model inheritance when available.
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    try:
        from openenv_core.env_server.types import Action, Observation, State
    except ImportError:
        class Action(BaseModel):
            pass

        class Observation(BaseModel):
            pass

        class State(BaseModel):
            pass


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class AnomalyType(str, Enum):
    REPEAT_VIOLATOR = "repeat_violator"
    HIGH_RISK_ORIGIN = "high_risk_origin"
    WEIGHT_VOLUME_MISMATCH = "weight_volume_mismatch"
    SEVERE_UNDERVALUATION = "severe_undervaluation"
    NEW_IEC_HIGH_VALUE = "new_iec_high_value"
    SUSPICIOUS_ROUTING = "suspicious_routing"
    UNDISCLOSED_RELATED_PARTY = "undisclosed_related_party"
    HS_CODE_RISK = "hs_code_risk"


class Channel(str, Enum):
    GREEN = "GREEN"
    ORANGE = "ORANGE"
    RED = "RED"


# ---------------------------------------------------------------------------
# Severity weights (Issue #5 fix: correct labels in comments)
# CRITICAL: 1.4–1.5 | HIGH: 1.3 | MEDIUM: 1.2 | LOW: 1.1 | BASE: 1.0
# ---------------------------------------------------------------------------

ANOMALY_SEVERITY: Dict[AnomalyType, float] = {
    AnomalyType.REPEAT_VIOLATOR: 1.5,          # CRITICAL
    AnomalyType.HIGH_RISK_ORIGIN: 1.4,         # CRITICAL
    AnomalyType.WEIGHT_VOLUME_MISMATCH: 1.3,   # HIGH
    AnomalyType.SEVERE_UNDERVALUATION: 1.3,    # HIGH
    AnomalyType.NEW_IEC_HIGH_VALUE: 1.2,       # MEDIUM
    AnomalyType.SUSPICIOUS_ROUTING: 1.2,       # MEDIUM
    AnomalyType.UNDISCLOSED_RELATED_PARTY: 1.1,# LOW
    AnomalyType.HS_CODE_RISK: 1.0,             # BASE
}

# Keywords used by SCNGrader to check anomaly coverage in notice text
ANOMALY_KEYWORDS: Dict[AnomalyType, List[str]] = {
    AnomalyType.REPEAT_VIOLATOR: ["repeat", "prior violation", "previous violation", "section 127"],
    AnomalyType.HIGH_RISK_ORIGIN: ["high risk", "FATF", "sanctioned", "grey list", "blacklist"],
    AnomalyType.WEIGHT_VOLUME_MISMATCH: ["weight", "volume", "mismatch", "capacity", "density"],
    AnomalyType.SEVERE_UNDERVALUATION: ["undervaluat", "declared value", "market value", "section 14"],
    AnomalyType.NEW_IEC_HIGH_VALUE: ["IEC", "new importer", "first shipment", "high value"],
    AnomalyType.SUSPICIOUS_ROUTING: ["routing", "transshipment", "irrational", "suspicious route"],
    AnomalyType.UNDISCLOSED_RELATED_PARTY: ["related party", "buyer-seller", "undisclosed", "Rule 3"],
    AnomalyType.HS_CODE_RISK: ["HS code", "HSN", "misdeclaration", "classification"],
}

# Valid Customs Act 1962 section numbers for SCN grading
# Issue #4 fix: Section 18 added (Provisional Assessment of Duty)
VALID_CUSTOMS_SECTIONS = {14, 18, 46, 47, 111, 112, 113, 114, 127}
VALID_CUSTOMS_SECTIONS_STR = {"114A"}  # string sections


# ---------------------------------------------------------------------------
# Domain data models
# ---------------------------------------------------------------------------

class CargoManifest(BaseModel):
    """A Bill of Entry / cargo manifest as seen by a customs officer."""
    boe_number: str
    port_of_entry: str
    importer_name: str
    iec_code: str
    iec_age_months: int                     # how old is the IEC?
    country_of_origin: str
    country_of_export: str
    commodity: str
    hs_code: str
    declared_weight_kg: float
    declared_value_usd: float
    market_value_usd: Optional[float] = None
    previous_violations: int = 0
    related_party: bool = False
    related_party_disclosed: bool = True
    routing_countries: List[str] = []
    container_type: str = "20GP"            # container type for density check
    description: str = ""


class CaseMetadata(BaseModel):
    """Ground truth metadata for a synthetic case."""
    case_id: str
    difficulty: str                         # clean / easy / medium / hard
    true_anomalies: List[AnomalyType]
    correct_channel: Channel
    duty_gap_inr: Optional[float] = None    # estimated duty evasion in INR


class CustomsCase(BaseModel):
    """A complete synthetic customs case."""
    manifest: CargoManifest
    metadata: CaseMetadata


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    # Issue #1 fix: task_name is Optional with default so HEALTHCHECK works
    task_name: str = "manifest-anomaly-detection"
    case_id: Optional[str] = None


class StepRequest(BaseModel):
    task: str
    anomalies: Optional[List[str]] = None
    channel: Optional[str] = None
    notice_text: Optional[str] = None       # SCN text from agent


class CustomsAction(Action):
    """OpenEnv-compatible action model for this environment."""

    task: str
    anomalies: Optional[List[str]] = None
    channel: Optional[str] = None
    notice_text: Optional[str] = None


class CustomsObservation(Observation):
    """OpenEnv-compatible observation model."""

    task_name: Optional[str] = None
    step: int = 0
    max_steps: int = 0
    manifest: Optional[CargoManifest] = None
    feedback: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    done: bool = False


class CustomsState(State):
    """OpenEnv-compatible environment state model."""

    episode_id: Optional[str] = None
    step_count: int = 0
    task_name: Optional[str] = None
    max_steps: int = 0
    done: bool = True
    cumulative_reward: float = 0.0


class ResetResponse(BaseModel):
    episode_id: str
    task_name: str
    manifest: CargoManifest
    step: int
    max_steps: int


class StepResponse(BaseModel):
    reward: float
    feedback: str
    details: Dict[str, Any]
    done: bool
    step: int
    cumulative_reward: float


class EnvironmentState(BaseModel):
    episode_id: Optional[str]
    task_name: Optional[str]
    step: int
    max_steps: int
    done: bool
    cumulative_reward: float
    manifest: Optional[CargoManifest]

