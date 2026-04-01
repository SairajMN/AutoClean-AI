"""
OpenEnv Data Cleaning Environment - Action Execution Engine
Validates, routes, and executes actions with rollback support.
All transitions are deterministic.
"""

import copy
import logging
from typing import Dict, Any, List, Optional, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger("openenv-datacleaner.action_engine")


class ActionValidationError(Exception):
    """Raised when an action fails validation."""
    pass


class ActionExecutionError(Exception):
    """Raised when an action fails during execution."""
    pass


class ActionEngine:
    """
    Deterministic action execution engine for data cleaning operations.
    Supports validation, dispatch, execution, and rollback.
    """

    def __init__(self):
        self._dataset: Optional[pd.DataFrame] = None
        self._original_dataset: Optional[pd.DataFrame] = None
        self._action_history: List[Dict[str, Any]] = []
        self._dataset_snapshots: List[pd.DataFrame] = []
        self._registered_actions: Dict[str, Callable] = {}
        self._register_actions()

    def _register_actions(self):
        """Register all available data cleaning actions."""
        self._registered_actions = {
            "drop_nulls": self._action_drop_nulls,
            "fill_nulls": self._action_fill_nulls,
            "remove_duplicates": self._action_remove_duplicates,
            "filter_rows": self._action_filter_rows,
            "drop_columns": self._action_drop_columns,
            "rename_columns": self._action_rename_columns,
            "convert_types": self._action_convert_types,
            "validate_email": self._action_validate_email,
            "outlier_removal": self._action_outlier_removal,
            "normalize": self._action_normalize,
        }

    def set_dataset(self, dataset: pd.DataFrame):
        """Set the current dataset and store original copy."""
        self._dataset = dataset.copy()
        self._original_dataset = dataset.copy()
        self._action_history = []
        self._dataset_snapshots = []

    @property
    def dataset(self) -> Optional[pd.DataFrame]:
        """Get current dataset."""
        return self._dataset

    @property
    def original_dataset(self) -> Optional[pd.DataFrame]:
        """Get original dataset."""
        return self._original_dataset

    @property
    def action_history(self) -> List[Dict[str, Any]]:
        """Get action history."""
        return list(self._action_history)

    def get_available_actions(self) -> List[str]:
        """Get list of available action types."""
        return list(self._registered_actions.keys())

    def validate_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Validate an action before execution.
        Returns True if valid, raises ActionValidationError otherwise.
        """
        if action_type not in self._registered_actions:
            raise ActionValidationError(
                f"Unknown action type: '{action_type}'. "
                f"Available: {self.get_available_actions()}"
            )

        if self._dataset is None:
            raise ActionValidationError("No dataset loaded. Reset environment first.")

        # Action-specific validation
        if action_type == "drop_columns":
            columns = params.get("columns", [])
            if isinstance(columns, str):
                columns = [columns]
            missing = [c for c in columns if c not in self._dataset.columns]
            if missing:
                raise ActionValidationError(f"Columns not found: {missing}")

        elif action_type in ("filter_rows", "convert_types", "outlier_removal", "normalize"):
            column = params.get("column")
            if column and column not in self._dataset.columns:
                raise ActionValidationError(f"Column '{column}' not found")

        elif action_type == "fill_nulls":
            strategy = params.get("strategy", "mean")
            valid_strategies = ["mean", "median", "mode", "value", "forward_fill", "backward_fill"]
            if strategy not in valid_strategies:
                raise ActionValidationError(
                    f"Invalid strategy: '{strategy}'. Valid: {valid_strategies}"
                )

        elif action_type == "filter_rows":
            operator = params.get("operator", "==")
            valid_ops = ["==", "!=", ">", "<", ">=", "<=", "contains", "notnull", "isnull"]
            if operator not in valid_ops:
                raise ActionValidationError(
                    f"Invalid operator: '{operator}'. Valid: {valid_ops}"
                )

        return True

    def execute_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and execute an action.
        Saves snapshot for rollback support.
        Returns action result dict.
        """
        # Validate
        self.validate_action(action_type, params)

        # Save snapshot for rollback
        self._dataset_snapshots.append(self._dataset.copy())

        # Execute
        handler = self._registered_actions[action_type]
        try:
            result = handler(params)
        except Exception as e:
            # Rollback on failure
            self._dataset = self._dataset_snapshots.pop()
            raise ActionExecutionError(f"Action '{action_type}' failed: {str(e)}")

        # Record in history
        self._action_history.append({
            "action_type": action_type,
            "params": params,
            "result": result
        })

        logger.info(f"Executed action: {action_type} with params: {params}")
        return result

    def revert_last_action(self) -> bool:
        """
        Revert the last executed action.
        Returns True if revert was successful, False if no actions to revert.
        """
        if not self._action_history or not self._dataset_snapshots:
            return False

        self._dataset = self._dataset_snapshots.pop()
        reverted = self._action_history.pop()
        logger.info(f"Reverted action: {reverted['action_type']}")
        return True

    def reset(self):
        """Reset engine to original dataset state."""
        if self._original_dataset is not None:
            self._dataset = self._original_dataset.copy()
        self._action_history = []
        self._dataset_snapshots = []

    # ============================================================
    # Action Implementations
    # ============================================================

    def _action_drop_nulls(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Drop rows with null values."""
        column = params.get("column")
        initial_shape = self._dataset.shape

        if column and column in self._dataset.columns:
            self._dataset = self._dataset.dropna(subset=[column])
        else:
            self._dataset = self._dataset.dropna()

        rows_removed = initial_shape[0] - self._dataset.shape[0]
        return {
            "action": "drop_nulls",
            "rows_removed": rows_removed,
            "new_shape": list(self._dataset.shape)
        }

    def _action_fill_nulls(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill null values with specified strategy."""
        column = params.get("column")
        strategy = params.get("strategy", "mean")
        value = params.get("value")

        columns = [column] if column else self._dataset.columns.tolist()
        filled_count = 0

        for col in columns:
            if col not in self._dataset.columns:
                continue

            null_count = int(self._dataset[col].isnull().sum())
            if null_count == 0:
                continue

            if strategy == "mean" and pd.api.types.is_numeric_dtype(self._dataset[col]):
                self._dataset[col] = self._dataset[col].fillna(self._dataset[col].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(self._dataset[col]):
                self._dataset[col] = self._dataset[col].fillna(self._dataset[col].median())
            elif strategy == "mode":
                mode_val = self._dataset[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else ""
                self._dataset[col] = self._dataset[col].fillna(fill_val)
            elif strategy == "value" and value is not None:
                self._dataset[col] = self._dataset[col].fillna(value)
            elif strategy == "forward_fill":
                self._dataset[col] = self._dataset[col].ffill()
            elif strategy == "backward_fill":
                self._dataset[col] = self._dataset[col].bfill()
            else:
                self._dataset[col] = self._dataset[col].fillna("")

            filled_count += null_count

        return {
            "action": "fill_nulls",
            "values_filled": filled_count,
            "strategy": strategy
        }

    def _action_remove_duplicates(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate rows."""
        initial_shape = self._dataset.shape
        subset = params.get("columns")

        self._dataset = self._dataset.drop_duplicates(subset=subset)

        rows_removed = initial_shape[0] - self._dataset.shape[0]
        return {
            "action": "remove_duplicates",
            "rows_removed": rows_removed,
            "new_shape": list(self._dataset.shape)
        }

    def _action_filter_rows(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter rows based on condition."""
        column = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")

        if column not in self._dataset.columns:
            raise ActionExecutionError(f"Column '{column}' not found")

        initial_count = len(self._dataset)

        if operator == "==":
            self._dataset = self._dataset[self._dataset[column] == value]
        elif operator == "!=":
            self._dataset = self._dataset[self._dataset[column] != value]
        elif operator == ">":
            self._dataset = self._dataset[self._dataset[column] > value]
        elif operator == "<":
            self._dataset = self._dataset[self._dataset[column] < value]
        elif operator == ">=":
            self._dataset = self._dataset[self._dataset[column] >= value]
        elif operator == "<=":
            self._dataset = self._dataset[self._dataset[column] <= value]
        elif operator == "contains":
            self._dataset = self._dataset[
                self._dataset[column].astype(str).str.contains(str(value), na=False)
            ]
        elif operator == "notnull":
            self._dataset = self._dataset[self._dataset[column].notna()]
        elif operator == "isnull":
            self._dataset = self._dataset[self._dataset[column].isna()]
        else:
            raise ActionExecutionError(f"Unknown operator: {operator}")

        rows_filtered = initial_count - len(self._dataset)
        return {
            "action": "filter_rows",
            "rows_filtered": rows_filtered,
            "remaining": len(self._dataset)
        }

    def _action_drop_columns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Drop specified columns."""
        columns = params.get("columns", [])
        if isinstance(columns, str):
            columns = [columns]

        existing_cols = [c for c in columns if c in self._dataset.columns]
        self._dataset = self._dataset.drop(columns=existing_cols, errors="ignore")

        return {
            "action": "drop_columns",
            "columns_dropped": existing_cols,
            "remaining_columns": self._dataset.columns.tolist()
        }

    def _action_rename_columns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rename columns."""
        mapping = params.get("mapping", {})
        self._dataset = self._dataset.rename(columns=mapping)
        return {
            "action": "rename_columns",
            "renamed": mapping,
            "columns": self._dataset.columns.tolist()
        }

    def _action_convert_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert column data types."""
        column = params.get("column")
        dtype = params.get("dtype", "str")

        if column and column in self._dataset.columns:
            try:
                if dtype == "int":
                    self._dataset[column] = pd.to_numeric(
                        self._dataset[column], errors="coerce"
                    ).astype("Int64")
                elif dtype == "float":
                    self._dataset[column] = pd.to_numeric(
                        self._dataset[column], errors="coerce"
                    )
                elif dtype == "str":
                    self._dataset[column] = self._dataset[column].astype(str)
                elif dtype == "datetime":
                    self._dataset[column] = pd.to_datetime(
                        self._dataset[column], errors="coerce"
                    )
                elif dtype == "bool":
                    self._dataset[column] = self._dataset[column].astype(bool)
            except Exception as e:
                raise ActionExecutionError(
                    f"Type conversion failed for {column}: {str(e)}"
                )

        return {
            "action": "convert_types",
            "column": column,
            "dtype": dtype
        }

    def _action_validate_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate email format in specified column."""
        column = params.get("column", "email")

        if column not in self._dataset.columns:
            raise ActionExecutionError(f"Column '{column}' not found")

        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        valid_mask = self._dataset[column].astype(str).str.match(
            email_pattern, na=False
        )
        invalid_count = int((~valid_mask).sum())

        if params.get("drop_invalid", False):
            self._dataset = self._dataset[valid_mask]

        return {
            "action": "validate_email",
            "column": column,
            "valid_count": int(valid_mask.sum()),
            "invalid_count": invalid_count,
            "dropped": params.get("drop_invalid", False)
        }

    def _action_outlier_removal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove outliers using IQR method."""
        column = params.get("column")
        multiplier = params.get("multiplier", 1.5)

        if (
            column
            and column in self._dataset.columns
            and pd.api.types.is_numeric_dtype(self._dataset[column])
        ):
            Q1 = self._dataset[column].quantile(0.25)
            Q3 = self._dataset[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            initial_count = len(self._dataset)
            self._dataset = self._dataset[
                (self._dataset[column] >= lower_bound)
                & (self._dataset[column] <= upper_bound)
            ]
            outliers_removed = initial_count - len(self._dataset)
        else:
            outliers_removed = 0

        return {
            "action": "outlier_removal",
            "outliers_removed": outliers_removed,
            "column": column
        }

    def _action_normalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize numeric columns."""
        column = params.get("column")
        method = params.get("method", "minmax")

        if (
            column
            and column in self._dataset.columns
            and pd.api.types.is_numeric_dtype(self._dataset[column])
        ):
            if method == "minmax":
                min_val = self._dataset[column].min()
                max_val = self._dataset[column].max()
                if max_val != min_val:
                    self._dataset[column] = (
                        self._dataset[column] - min_val
                    ) / (max_val - min_val)
            elif method == "zscore":
                mean = self._dataset[column].mean()
                std = self._dataset[column].std()
                if std != 0:
                    self._dataset[column] = (
                        self._dataset[column] - mean
                    ) / std

        return {
            "action": "normalize",
            "column": column,
            "method": method
        }