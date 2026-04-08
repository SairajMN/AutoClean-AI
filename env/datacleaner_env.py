"""
OpenEnv Data Cleaning Environment - Core Environment Implementation
Extends Environment from openenv-core for full OpenEnv lifecycle compliance.
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

from openenv.core import Environment, Action, Observation, State

try:
    from .models import (
        DataCleaningAction,
        DataCleaningObservation,
        DataCleaningState,
        Reward,
        GradeResult,
    )
    from .action_engine import ActionEngine, ActionValidationError, ActionExecutionError
    from .tasks import get_task_config, get_tasks
    from .grader import Grader
    from .reward import RewardCalculator
except ImportError:  # pragma: no cover - supports direct execution from env/
    from models import (
        DataCleaningAction,
        DataCleaningObservation,
        DataCleaningState,
        Reward,
        GradeResult,
    )
    from action_engine import ActionEngine, ActionValidationError, ActionExecutionError
    from tasks import get_task_config, get_tasks
    from grader import Grader
    from reward import RewardCalculator

logger = logging.getLogger("openenv-datacleaner.env")


class DataCleaningEnv(Environment):
    """
    OpenEnv-compliant Data Cleaning Environment.
    
    Implements the full OpenEnv lifecycle:
    - reset(seed, episode_id): Initialize environment for a task
    - step(action): Execute an action and return observation
    - state: Return current environment state
    
    This environment is deterministic and production-ready.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the DataCleaningEnv."""
        super().__init__()
        
        # Core components
        self._action_engine = ActionEngine()
        self._grader = Grader()
        self._reward_calculator = RewardCalculator()
        
        # State tracking
        self._session_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._grade_result: Optional[GradeResult] = None
        self._episode_id: Optional[str] = None
        
        # Dataset storage
        self._dataset: Optional[pd.DataFrame] = None
        self._original_dataset: Optional[pd.DataFrame] = None
        
        logger.info("DataCleaningEnv initialized")

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        """
        Reset environment and initialize a new task.
        OpenEnv lifecycle method.
        
        Args:
            seed: Optional seed for deterministic behavior
            episode_id: Optional episode identifier
            **kwargs: Additional arguments (task_id, session_id)
            
        Returns:
            DataCleaningObservation: Initial observation with dataset info
        """
        task_id = kwargs.get("task_id", "easy_001")
        session_id = kwargs.get("session_id", str(uuid.uuid4()))
        
        logger.info(f"Resetting environment: task_id={task_id}, episode_id={episode_id}")
        
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Validate task
        available_tasks = get_tasks()
        if task_id not in available_tasks:
            raise ValueError(
                f"Invalid task_id: '{task_id}'. Available: {available_tasks}"
            )
        
        # Reset state
        self._task_id = task_id
        self._session_id = session_id
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._grade_result = None
        
        # Get task config
        task_config = get_task_config(task_id)
        
        # Generate dataset for task
        self._generate_dataset(task_config.dataset_config)
        
        # Set up action engine
        self._action_engine.set_dataset(self._dataset)
        
        # Set up reward calculator
        self._reward_calculator.setup(
            original_dataset=self._original_dataset,
            expected_actions=task_config.expected_actions
        )
        
        # Build initial observation
        observation = self._build_observation(
            message=f"Task '{task_id}' initialized. Difficulty: {task_config.difficulty}"
        )
        
        logger.info(
            f"Environment reset complete. Episode: {self._episode_id}, "
            f"Task: {task_id}, Dataset shape: {self._dataset.shape}"
        )
        
        return observation

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DataCleaningObservation:
        """
        Take a step in the environment.
        OpenEnv lifecycle method.
        
        Args:
            action: Action to execute (DataCleaningAction)
            timeout_s: Optional timeout
            **kwargs: Additional arguments
            
        Returns:
            DataCleaningObservation with reward and done status
        """
        # Check if episode is done
        if self._done:
            obs = self._build_observation(
                message="Episode is done. Reset to start a new episode.",
                done=True
            )
            obs.reward = 0.0
            return obs
        
        # Extract action type from action metadata
        action_metadata = action.metadata if action.metadata else {}
        action_type = action_metadata.get("action_type", "")
        params = action_metadata.get("params", {})
        
        # Increment step count
        self._step_count += 1
        
        # Check for submit action
        if action_type == "submit":
            return self._handle_submit()
        
        # Check for revert action
        if action_type == "revert":
            return self._handle_revert()
        
        # Execute action through engine
        try:
            result = self._action_engine.execute_action(
                action_type=action_type,
                params=params
            )
        except ActionValidationError as e:
            observation = self._build_observation(
                message=f"Action validation failed: {str(e)}",
                done=False
            )
            observation.reward = 0.0
            return observation

        except ActionExecutionError as e:
            observation = self._build_observation(
                message=f"Action execution failed: {str(e)}",
                done=False
            )
            observation.reward = 0.0
            return observation
        
        # Calculate reward
        reward_obj = self._reward_calculator.calculate(
            action_type=action_type,
            params=params,
            current_dataset=self._action_engine.dataset,
            step_count=self._step_count
        )
        
        # Build observation
        observation = self._build_observation(
            message=f"Action '{action_type}' executed successfully",
            done=False
        )
        observation.reward = reward_obj.value
        
        logger.info(
            f"Step {self._step_count}: {action_type} -> "
            f"reward={reward_obj.value:.4f}"
        )
        
        return observation

    @property
    def state(self) -> DataCleaningState:
        """
        Get the current environment state.
        OpenEnv lifecycle method.
        
        Returns:
            DataCleaningState: Current state snapshot
        """
        dataset_info = {}
        if self._action_engine.dataset is not None:
            dataset_info = {
                "shape": list(self._action_engine.dataset.shape),
                "columns": self._action_engine.dataset.columns.tolist(),
                "null_counts": {
                    col: int(self._action_engine.dataset[col].isnull().sum())
                    for col in self._action_engine.dataset.columns
                }
            }
        
        return DataCleaningState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            session_id=self._session_id or "",
            task_id=self._task_id,
            action_history=self._action_engine.action_history,
            grade=self._grade_result.model_dump() if self._grade_result else None
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get environment metadata."""
        return {
            "name": "openenv-datacleaner",
            "version": "1.0.0",
            "tasks": get_tasks(),
            "available_actions": self._action_engine.get_available_actions()
        }

    # ============================================================
    # Internal Methods
    # ============================================================

    def _generate_dataset(self, config: Dict[str, Any]):
        """Generate a dataset based on task configuration."""
        rows = config.get("rows", 100)
        columns = config.get("columns", ["id", "name", "age", "email", "salary"])
        null_pct = config.get("null_percentage", 0.1)
        dup_count = config.get("duplicate_count", 5)
        
        # Set seed for deterministic generation
        np.random.seed(42 + hash(self._task_id or "") % 1000)
        
        data = {}
        
        for col in columns:
            if col == "id":
                data[col] = list(range(1, rows + 1))
            elif col == "name":
                data[col] = [
                    f"Person_{i}" if np.random.random() > null_pct else None
                    for i in range(rows)
                ]
            elif col == "age":
                ages = np.random.randint(18, 80, rows).astype(float)
                null_indices = np.random.choice(rows, size=int(rows * null_pct), replace=False)
                ages[null_indices] = np.nan
                data[col] = ages
            elif col == "email":
                emails = []
                for i in range(rows):
                    if np.random.random() > null_pct:
                        if np.random.random() > 0.15:
                            emails.append(f"user{i}@example.com")
                        else:
                            emails.append(f"invalid-email-{i}")
                    else:
                        emails.append(None)
                data[col] = emails
            elif col == "salary":
                salaries = np.random.uniform(30000, 150000, rows)
                null_indices = np.random.choice(rows, size=int(rows * null_pct), replace=False)
                salaries[null_indices] = np.nan
                data[col] = salaries
            elif col == "department":
                depts = ["Engineering", "Sales", "Marketing", "HR", None]
                data[col] = np.random.choice(depts, rows)
            elif col == "join_date":
                dates = pd.date_range("2020-01-01", periods=rows, freq="D")
                data[col] = dates
            elif col == "score":
                scores = np.random.normal(75, 15, rows)
                null_indices = np.random.choice(rows, size=int(rows * null_pct), replace=False)
                scores[null_indices] = np.nan
                data[col] = scores
            else:
                data[col] = np.random.randn(rows)
        
        self._dataset = pd.DataFrame(data)
        
        # Introduce duplicates
        if dup_count > 0 and len(self._dataset) > dup_count:
            dup_indices = np.random.choice(len(self._dataset), size=dup_count, replace=False)
            for idx in dup_indices:
                if idx > 0:
                    self._dataset.iloc[idx] = self._dataset.iloc[idx - 1].copy()
        
        self._original_dataset = self._dataset.copy()
        
        # Reset seed
        np.random.seed(None)

    def _build_observation(
        self,
        message: str = "",
        done: bool = False
    ) -> DataCleaningObservation:
        """Build a DataCleaningObservation from current state."""
        dataset_info = {}
        if self._action_engine.dataset is not None:
            dataset_info = {
                "shape": list(self._action_engine.dataset.shape),
                "columns": self._action_engine.dataset.columns.tolist(),
                "null_counts": {
                    col: int(self._action_engine.dataset[col].isnull().sum())
                    for col in self._action_engine.dataset.columns
                },
                "dtypes": {
                    col: str(dtype)
                    for col, dtype in self._action_engine.dataset.dtypes.items()
                }
            }

        return DataCleaningObservation(
            dataset_info=dataset_info,
            available_actions=self._action_engine.get_available_actions() + ["submit", "revert"],
            step_count=self._step_count,
            task_id=self._task_id,
            message=message,
            done=done or self._done,
            reward=None,
            metadata={
                "dataset_info": dataset_info,
                "available_actions": self._action_engine.get_available_actions() + ["submit", "revert"],
                "step_count": self._step_count,
                "task_id": self._task_id,
                "message": message,
                "session_id": self._session_id,
                "episode_id": self._episode_id,
            }
        )

    def _handle_submit(self) -> DataCleaningObservation:
        """Handle submit action - grade the solution."""
        self._done = True
        
        # Grade the solution
        self._grader.setup(
            task_id=self._task_id,
            original_dataset=self._original_dataset,
            current_dataset=self._action_engine.dataset,
            action_history=self._action_engine.action_history
        )
        
        self._grade_result = self._grader.grade()
        
        # Calculate terminal reward
        reward_obj = self._reward_calculator.calculate_terminal_reward(
            current_dataset=self._action_engine.dataset,
            grade_result={
                "final_score": self._grade_result.final_score,
                "breakdown": self._grade_result.breakdown
            }
        )
        
        observation = self._build_observation(
            message=f"Submitted! Score: {self._grade_result.final_score:.2f}",
            done=True
        )
        observation.reward = reward_obj.value
        
        logger.info(
            f"Submitted! Task: {self._task_id}, "
            f"Score: {self._grade_result.final_score:.4f}"
        )
        
        return observation

    def _handle_revert(self) -> DataCleaningObservation:
        """Handle revert action - undo last action."""
        reverted = self._action_engine.revert_last_action()
        
        observation = self._build_observation(
            message="Last action reverted" if reverted else "Nothing to revert",
            done=False
        )
        observation.reward = 0.0

        return observation

    # ============================================================
    # Utility Methods
    # ============================================================

    def get_dataset(self) -> Optional[pd.DataFrame]:
        """Get current dataset (for internal use)."""
        return self._action_engine.dataset

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history."""
        return self._action_engine.action_history

    def is_done(self) -> bool:
        """Check if episode is done."""
        return self._done

    def get_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id

    def get_task_id(self) -> Optional[str]:
        """Get current task ID."""
        return self._task_id
