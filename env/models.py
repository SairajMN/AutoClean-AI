# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data Cleaning Environment.
Uses openenv-core base classes for Action, Observation, and State.
"""

from typing import Optional, Dict, Any, List

from openenv.core import Action, Observation, State
from pydantic import BaseModel, Field


# ============================================================
# Original Env Environment Models
# ============================================================


class EnvAction(Action):
    """Action for the Env environment - just a message to echo."""

    message: str = Field(default="", description="Message to echo back")


class EnvObservation(Observation):
    """Observation from the Env environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")


# ============================================================
# Data Cleaning Environment Models
# ============================================================


class DataCleaningAction(Action):
    """
    OpenEnv-compliant action model for data cleaning.
    Represents a single action to be executed in the environment.
    """
    action_type: str = Field(
        default="",
        description="Type of action to execute (e.g., 'drop_nulls', 'fill_nulls')"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Associated task ID"
    )


class DataCleaningObservation(Observation):
    """
    OpenEnv-compliant observation model for data cleaning.
    Represents the state observation returned after reset or step.
    """
    dataset_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current dataset metadata"
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of valid actions"
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Current task ID"
    )
    message: str = Field(
        default="",
        description="Status message"
    )


class DataCleaningState(State):
    """
    Complete environment state for serialization.
    """
    session_id: str = Field(default="")
    task_id: Optional[str] = Field(default=None)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_hash: Optional[str] = Field(default=None)
    grade: Optional[Dict[str, Any]] = Field(default=None)


# ============================================================
# Supporting Data Models (not inheriting from openenv-core)
# ============================================================


class Reward(BaseModel):
    """
    Structured reward with components for quality, progress, and penalties.
    """
    value: float = Field(
        default=0.0,
        description="Total reward value"
    )
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward components"
    )

    @classmethod
    def create(
        cls,
        quality: float = 0.0,
        progress: float = 0.0,
        penalty: float = 0.0
    ) -> "Reward":
        """Factory method to create a structured reward."""
        value = quality + progress - penalty
        return cls(
            value=round(value, 4),
            components={
                "quality": round(quality, 4),
                "progress": round(progress, 4),
                "penalty": round(penalty, 4)
            }
        )


class TaskConfig(BaseModel):
    """
    Configuration for a data cleaning task.
    """
    task_id: str = Field(
        ...,
        description="Unique task identifier"
    )
    difficulty: str = Field(
        ...,
        description="Task difficulty level (easy, medium, hard)"
    )
    description: str = Field(
        default="",
        description="Task description"
    )
    dataset_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dataset generation configuration"
    )
    expected_actions: List[str] = Field(
        default_factory=list,
        description="Expected sequence of actions for optimal solution"
    )
    grading_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Criteria for grading the task"
    )


class GradeResult(BaseModel):
    """
    Result from grading a submitted solution.
    """
    final_score: float = Field(
        default=0.0,
        description="Final score (0.0 to 1.0)"
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Score breakdown by criterion"
    )
    feedback: str = Field(
        default="",
        description="Feedback on the solution"
    )