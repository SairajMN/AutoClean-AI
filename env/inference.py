#!/usr/bin/env python3
"""
Baseline Inference Script for OpenEnv Data Cleaner
Uses OpenAI API client to run an LLM agent against the data cleaning environment.
Produces reproducible baseline scores on all 3 tasks.

Environment Variables:
    API_BASE_URL - The API endpoint for the LLM
    MODEL_NAME   - The model identifier to use for inference
    HF_TOKEN     - Your Hugging Face / API key
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional

from openai import OpenAI

# ============================================================
# Configuration
# ============================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "dummy-key"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")
MAX_STEPS = 20
TASKS = ["easy_001", "medium_001", "hard_001"]

# ============================================================
# Logging Helpers
# ============================================================

def log_start(task: str, env: str, model: str) -> None:
    """Log the start of a task run."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Log a single step."""
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={done}", flush=True)
    if error:
        print(f"[ERROR] {error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log the end of a task run."""
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={json.dumps(rewards)}", flush=True)


# ============================================================
# Environment Client
# ============================================================

class DataCleaningEnvClient:
    """HTTP client for the Data Cleaning Environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None
        self._task_id: Optional[str] = None

    async def reset(self, task_id: str = "easy_001") -> Dict[str, Any]:
        """Reset the environment and start a new task."""
        import aiohttp
        url = f"{self.base_url}/reset"
        payload = {"task_id": task_id}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
        self._task_id = task_id
        self._session_id = result.get("session_id")
        return result

    async def step(self, action_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an action in the environment."""
        import aiohttp
        url = f"{self.base_url}/step"
        payload = {
            "action_type": action_type,
            "params": params or {}
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
        return result

    async def submit(self) -> Dict[str, Any]:
        """Submit the current solution for grading."""
        import aiohttp
        url = f"{self.base_url}/submit"
        async with aiohttp.ClientSession() as session:
            async with session.post(url) as resp:
                result = await resp.json()
        return result

    async def get_tasks(self) -> List[Dict[str, Any]]:
        """Get available tasks."""
        import aiohttp
        url = f"{self.base_url}/tasks"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                result = await resp.json()
        return result.get("tasks", [])

    async def get_dataset(self) -> Dict[str, Any]:
        """Get current dataset information."""
        import aiohttp
        url = f"{self.base_url}/dataset"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                result = await resp.json()
        return result


# ============================================================
# LLM Agent
# ============================================================

def get_system_prompt(dataset_info: Dict[str, Any], task_info: Dict[str, Any] = None) -> str:
    """Generate system prompt for the data cleaning agent."""
    prompt = """You are an AI data cleaning agent. Your goal is to clean and prepare a dataset.

Available actions:
- drop_nulls: Remove rows with null values (params: column - optional)
- fill_nulls: Fill null values (params: column, strategy: mean/median/mode/forward_fill/backward_fill)
- remove_duplicates: Remove duplicate rows (params: columns - optional)
- filter_rows: Filter rows based on condition (params: column, operator, value)
- drop_columns: Remove columns (params: columns as comma-separated string)
- convert_types: Convert column data types (params: column, dtype: str/int/float/datetime)
- validate_email: Validate email format (params: column, drop_invalid: bool)
- outlier_removal: Remove outliers using IQR (params: column, multiplier: float)
- normalize: Normalize numeric columns (params: column, method: minmax/zscore)
- submit: Submit your solution for grading (no params)
- revert: Revert last action (no params)

Respond with ONLY a JSON object containing:
{
    "action_type": "the action to take",
    "params": {"param": "value"},
    "reasoning": "brief explanation of why"
}

Be efficient - use the minimum number of actions needed."""

    if dataset_info:
        prompt += f"\n\nCurrent dataset: {json.dumps(dataset_info, indent=2)}"

    if task_info:
        prompt += f"\n\nTask: {task_info.get('description', '')}"
        prompt += f"\nExpected actions: {task_info.get('expected_actions', [])}"

    return prompt


def get_model_message(client: OpenAI, step: int, dataset_info: Dict, task_info: Dict, history: List[str]) -> Dict[str, Any]:
    """Get action from the LLM model."""
    try:
        system_prompt = get_system_prompt(dataset_info, task_info)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add recent history (last 5 steps)
        for h in history[-5:]:
            messages.append({"role": "user", "content": h})

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )

        text = (completion.choices[0].message.content or "").strip()

        # Try to parse JSON response
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                action = json.loads(text[start:end])
                return action
        except json.JSONDecodeError:
            pass

        # Fallback: return submit if we've taken many steps
        if step >= MAX_STEPS:
            return {"action_type": "submit", "params": {}, "reasoning": "Max steps reached"}

        return {"action_type": "submit", "params": {}, "reasoning": "Could not parse model response"}

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "submit", "params": {}, "reasoning": f"Error: {exc}"}


# ============================================================
# Main Inference Loop
# ============================================================

async def run_task(env: DataCleaningEnvClient, task_id: str, client: OpenAI) -> Dict[str, Any]:
    """Run a single task and return results."""
    log_start(task=task_id, env="openenv-datacleaner", model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset environment
        result = await env.reset(task_id=task_id)
        dataset_info = result.get("observation", {})

        # Get task info
        tasks = await env.get_tasks()
        task_info = next((t for t in tasks if t.get("task_id") == task_id), {})

        for step in range(1, MAX_STEPS + 1):
            # Get action from model
            action = get_model_message(client, step, dataset_info, task_info, history)
            action_type = action.get("action_type", "submit")
            params = action.get("params", {})

            # Execute action
            if action_type == "submit":
                result = await env.submit()
            else:
                result = await env.step(action_type, params)

            reward = result.get("reward", 0.0) or 0.0
            done = result.get("done", False)
            obs = result.get("observation", {})

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done)

            history.append(f"Step {step}: {action_type} -> reward {reward:+.4f}")

            # Update dataset info
            dataset_info = obs

            if done:
                # Extract score from grade
                info = result.get("info", {})
                grade = info.get("grade", {})
                score = grade.get("final_score", 0.0)
                success = score >= 0.5
                break

        # If we didn't submit, do it now
        if not done:
            result = await env.submit()
            info = result.get("info", {})
            grade = info.get("grade", {})
            score = grade.get("final_score", 0.0)
            success = score >= 0.5
            rewards.append(result.get("reward", 0.0) or 0.0)
            steps_taken += 1

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return {"task_id": task_id, "score": 0.0, "success": False, "steps": steps_taken, "rewards": rewards}

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken, "rewards": rewards}


async def main() -> None:
    """Main entry point."""
    print(f"[INFO] Starting inference with model={MODEL_NAME}, base_url={API_BASE_URL}", flush=True)

    # Initialize clients
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataCleaningEnvClient(base_url=SPACE_URL)

    # Run all tasks
    results = []
    for task_id in TASKS:
        result = await run_task(env, task_id, client)
        results.append(result)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("[SUMMARY] Baseline Results", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task_id']}: score={r['score']:.4f} steps={r['steps']} [{status}]", flush=True)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average Score: {avg_score:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())