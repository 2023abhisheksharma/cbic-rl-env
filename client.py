"""CBIC Environment typed client.

This client supports OpenEnv EnvClient integrations when openenv/openenv-core
is installed, and also exposes HTTP convenience methods for this repository's
existing REST server endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Generic, Optional, TypeVar

import httpx

from environment.models import CustomsAction, CustomsObservation, CustomsState

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
StateT = TypeVar("StateT")


StepResult = None
EnvClient = None

for _prefix in ("openenv.core", "openenv_core"):
    try:
        _client_types = importlib.import_module(f"{_prefix}.client_types")
        _env_client = importlib.import_module(f"{_prefix}.env_client")
        StepResult = getattr(_client_types, "StepResult")
        EnvClient = getattr(_env_client, "EnvClient")
        break
    except Exception:
        continue

if StepResult is None or EnvClient is None:
    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: Optional[float] = None
        done: bool = False

    class EnvClient(Generic[ActT, ObsT, StateT]):
        """Small fallback to keep imports working without openenv installed."""

        def __init__(self, base_url: str, timeout: float = 60.0):
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout


class CbicEnv(EnvClient[CustomsAction, CustomsObservation, CustomsState]):
    """Typed client for the CBIC customs inspection environment."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        super().__init__(base_url=base_url, timeout=timeout)
        self._http = httpx.Client(timeout=timeout)

    def _step_payload(self, action: CustomsAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CustomsObservation]:
        obs_data = payload.get("observation")
        if not isinstance(obs_data, dict):
            obs_data = {
                "task_name": payload.get("task_name"),
                "step": payload.get("step", 0),
                "max_steps": payload.get("max_steps", 0),
                "manifest": payload.get("manifest"),
                "feedback": payload.get("feedback", ""),
                "details": payload.get("details", {}),
                "cumulative_reward": payload.get("cumulative_reward", 0.0),
                "done": payload.get("done", False),
            }

        obs = CustomsObservation(**obs_data)
        reward = payload.get("reward")
        done = bool(payload.get("done", False))
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> CustomsState:
        return CustomsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", payload.get("step", 0)),
            task_name=payload.get("task_name"),
            max_steps=payload.get("max_steps", 0),
            done=bool(payload.get("done", False)),
            cumulative_reward=float(payload.get("cumulative_reward", 0.0)),
        )

    # Convenience HTTP methods to work with the current REST API directly.
    def reset_http(self, task_name: str = "manifest-anomaly-detection", case_id: str | None = None) -> StepResult[CustomsObservation]:
        body: Dict[str, Any] = {"task_name": task_name}
        if case_id:
            body["case_id"] = case_id

        response = self._http.post(f"{self.base_url}/reset", json=body)
        response.raise_for_status()
        payload = response.json()
        return self._parse_result(payload)

    def step_http(self, action: CustomsAction) -> StepResult[CustomsObservation]:
        response = self._http.post(
            f"{self.base_url}/step",
            json=self._step_payload(action),
        )
        response.raise_for_status()
        payload = response.json()
        return self._parse_result(payload)

    def state_http(self) -> CustomsState:
        response = self._http.get(f"{self.base_url}/state")
        response.raise_for_status()
        payload = response.json()
        return self._parse_state(payload)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CbicEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
