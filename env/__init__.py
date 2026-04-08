"""OpenEnv Data Cleaning Environment."""

from .datacleaner_env import DataCleaningEnv
from .client import DataCleaningClient
from .models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    Reward,
    TaskConfig,
    GradeResult,
)

try:
    from .env import AutoCleanEnv as LegacyAutoCleanEnv
except Exception:  # pragma: no cover - keep package importable if legacy env breaks
    LegacyAutoCleanEnv = None

# Backward-compatible export for stale legacy runners.
AutoCleanEnv = LegacyAutoCleanEnv or DataCleaningEnv

__all__ = [
    "DataCleaningEnv",
    "AutoCleanEnv",
    "LegacyAutoCleanEnv",
    "DataCleaningClient",
    "DataCleaningAction",
    "DataCleaningObservation",
    "DataCleaningState",
    "Reward",
    "TaskConfig",
    "GradeResult",
]
