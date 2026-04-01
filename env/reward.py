"""
OpenEnv Data Cleaning Environment - Reward System
Computes structured rewards aligned with OpenEnv expectations.
"""

import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from env.models import Reward

logger = logging.getLogger("openenv-datacleaner.reward")


class RewardCalculator:
    """
    Calculates structured rewards for data cleaning actions.
    Reward = quality + progress - penalty
    """

    def __init__(self):
        self._original_dataset: Optional[pd.DataFrame] = None
        self._previous_dataset: Optional[pd.DataFrame] = None
        self._total_steps: int = 0
        self._expected_actions: List[str] = []
        self._actions_taken: List[str] = []

    def setup(
        self,
        original_dataset: pd.DataFrame,
        expected_actions: List[str]
    ):
        """Set up reward calculator with context."""
        self._original_dataset = original_dataset.copy()
        self._previous_dataset = original_dataset.copy()
        self._expected_actions = list(expected_actions)
        self._actions_taken = []
        self._total_steps = 0

    def calculate(
        self,
        action_type: str,
        params: Dict[str, Any],
        current_dataset: pd.DataFrame,
        step_count: int
    ) -> Reward:
        """
        Calculate reward for a single action.
        Returns Reward with value and components.
        """
        self._total_steps = step_count
        self._actions_taken.append(action_type)

        # Quality component: how much did this action improve data quality?
        quality = self._calculate_quality_reward(
            self._previous_dataset, current_dataset, action_type
        )

        # Progress component: how close to completing expected actions?
        progress = self._calculate_progress_reward()

        # Penalty component: inefficiency or harmful actions
        penalty = self._calculate_penalty(action_type, params)

        # Update previous dataset
        self._previous_dataset = current_dataset.copy()

        reward = Reward.create(
            quality=quality,
            progress=progress,
            penalty=penalty
        )

        logger.info(
            f"Reward for {action_type}: quality={quality:.4f}, "
            f"progress={progress:.4f}, penalty={penalty:.4f}, "
            f"total={reward.value:.4f}"
        )

        return reward

    def calculate_terminal_reward(
        self,
        current_dataset: pd.DataFrame,
        grade_result: Optional[Dict[str, Any]] = None
    ) -> Reward:
        """
        Calculate terminal reward at episode end.
        Based on final data quality and grade.
        """
        # Base quality from final dataset state
        quality = self._calculate_final_quality(current_dataset)

        # Progress based on expected actions completed
        progress = self._calculate_final_progress()

        # Penalty for excessive steps
        expected_count = len(self._expected_actions)
        actual_count = len(self._actions_taken)
        penalty = max(0.0, (actual_count - expected_count) * 0.05)

        # Bonus from grade if available
        if grade_result:
            grade_bonus = grade_result.get("final_score", 0.0) * 0.5
            quality += grade_bonus

        reward = Reward.create(
            quality=quality,
            progress=progress,
            penalty=penalty
        )

        return reward

    def _calculate_quality_reward(
        self,
        previous: pd.DataFrame,
        current: pd.DataFrame,
        action_type: str
    ) -> float:
        """Calculate quality improvement from action."""
        quality_score = 0.0

        if action_type == "drop_nulls":
            prev_nulls = previous.isnull().sum().sum()
            curr_nulls = current.isnull().sum().sum()
            if prev_nulls > 0:
                quality_score = (prev_nulls - curr_nulls) / prev_nulls * 0.3

        elif action_type == "fill_nulls":
            prev_nulls = previous.isnull().sum().sum()
            curr_nulls = current.isnull().sum().sum()
            if prev_nulls > 0:
                quality_score = (prev_nulls - curr_nulls) / prev_nulls * 0.3

        elif action_type == "remove_duplicates":
            prev_dups = previous.duplicated().sum()
            curr_dups = current.duplicated().sum()
            if prev_dups > 0:
                quality_score = (prev_dups - curr_dups) / prev_dups * 0.25

        elif action_type == "validate_email":
            quality_score = 0.15  # Fixed reward for validation

        elif action_type == "outlier_removal":
            quality_score = 0.1  # Fixed reward for outlier handling

        elif action_type == "convert_types":
            quality_score = 0.15  # Fixed reward for type conversion

        elif action_type == "normalize":
            quality_score = 0.1  # Fixed reward for normalization

        return round(min(max(quality_score, 0.0), 1.0), 4)

    def _calculate_progress_reward(self) -> float:
        """Calculate progress towards completing expected actions."""
        if not self._expected_actions:
            return 0.0

        completed = sum(
            1 for action in self._expected_actions
            if action in self._actions_taken
        )

        return round(completed / len(self._expected_actions) * 0.5, 4)

    def _calculate_penalty(self, action_type: str, params: Dict[str, Any]) -> float:
        """Calculate penalty for inefficient or harmful actions."""
        penalty = 0.0

        # Penalty for redundant actions
        if len(self._actions_taken) > 1:
            if self._actions_taken[-2] == action_type:
                penalty += 0.1

        # Penalty for excessive steps
        if self._total_steps > len(self._expected_actions) * 2:
            penalty += 0.05

        # Penalty for dropping too many rows
        if self._previous_dataset is not None and action_type in (
            "drop_nulls", "filter_rows", "outlier_removal"
        ):
            prev_len = len(self._previous_dataset)
            # This is calculated after the action, so we check current state
            # Penalty is applied in the action result, not here

        return round(min(max(penalty, 0.0), 1.0), 4)

    def _calculate_final_quality(self, current_dataset: pd.DataFrame) -> float:
        """Calculate final data quality score."""
        if self._original_dataset is None:
            return 0.0

        score = 0.0

        # Null handling (30%)
        orig_nulls = self._original_dataset.isnull().sum().sum()
        curr_nulls = current_dataset.isnull().sum().sum()
        if orig_nulls > 0:
            score += (1 - curr_nulls / orig_nulls) * 0.3
        else:
            score += 0.3

        # Duplicate handling (20%)
        orig_dups = self._original_dataset.duplicated().sum()
        curr_dups = current_dataset.duplicated().sum()
        if orig_dups > 0:
            score += (1 - curr_dups / orig_dups) * 0.2
        else:
            score += 0.2

        # Data completeness (25%)
        orig_rows = len(self._original_dataset)
        curr_rows = len(current_dataset)
        if orig_rows > 0:
            retention = curr_rows / orig_rows
            score += min(retention, 1.0) * 0.25

        # Column integrity (25%)
        orig_cols = set(self._original_dataset.columns)
        curr_cols = set(current_dataset.columns)
        if orig_cols:
            col_retention = len(orig_cols & curr_cols) / len(orig_cols)
            score += col_retention * 0.25

        return round(min(max(score, 0.0), 1.0), 4)

    def _calculate_final_progress(self) -> float:
        """Calculate final progress score."""
        if not self._expected_actions:
            return 0.0

        completed = sum(
            1 for action in self._expected_actions
            if action in self._actions_taken
        )

        return round(completed / len(self._expected_actions), 4)