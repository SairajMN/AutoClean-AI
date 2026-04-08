"""Compatibility wrapper for legacy runners importing `task` from repo root."""

from env.task import generate_task, get_task_description

__all__ = ["generate_task", "get_task_description"]
