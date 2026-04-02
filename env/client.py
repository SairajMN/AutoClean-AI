"""OpenEnv Data Cleaning Environment Client."""

from typing import Dict, Any, Optional

from openenv.core import EnvClient, SyncEnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DataCleaningAction, DataCleaningObservation, DataCleaningState


class DataCleaningClient(EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """
    Client for the Data Cleaning Environment.
    
    Example:
        >>> with DataCleaningClient(base_url="http://localhost:7860") as client:
        ...     result = client.reset(task_id="easy_001")
        ...     print(result.observation.metadata.get("message"))
    """

    def _step_payload(self, action: DataCleaningAction) -> Dict[str, Any]:
        """Convert DataCleaningAction to JSON payload."""
        return {
            "action_type": action.action_type,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataCleaningObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = DataCleaningObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data,
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleaningState:
        """Parse server response into State object."""
        return DataCleaningState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            session_id=payload.get("session_id", ""),
            task_id=payload.get("task_id"),
            action_history=payload.get("action_history", []),
            grade=payload.get("grade"),
        )