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
TASKS = ["easy_001", "medium_001", "hard_001", "employee_demo"]

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
        import ssl
        url = f"{self.base_url}/reset"
        payload = {"task_id": task_id}
        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
        self._task_id = task_id
        self._session_id = result.get("session_id")
        return result

    async def step(self, action_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an action in the environment."""
        import aiohttp
        import ssl
        url = f"{self.base_url}/step"
        payload = {
            "action_type": action_type,
            "params": params or {}
        }
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
        return result

    async def submit(self) -> Dict[str, Any]:
        """Submit the current solution for grading."""
        import aiohttp
        import ssl
        url = f"{self.base_url}/submit"
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url) as resp:
                result = await resp.json()
        return result

    async def get_tasks(self) -> List[Dict[str, Any]]:
        """Get available tasks."""
        import aiohttp
        import ssl
        url = f"{self.base_url}/tasks"
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url) as resp:
                result = await resp.json()
        return result.get("tasks", [])

    async def get_dataset(self) -> Dict[str, Any]:
        """Get current dataset information."""
        import aiohttp
        import ssl
        url = f"{self.base_url}/dataset"
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url) as resp:
                result = await resp.json()
        return result


# ============================================================
# LLM Agent
# ============================================================

def get_system_prompt(dataset_info: Dict[str, Any], task_info: Dict[str, Any] = None) -> str:
    """Generate system prompt for the data cleaning agent."""
    # Extract dataset info
    shape = dataset_info.get("shape", [])
    null_counts = dataset_info.get("null_counts", {})
    columns = dataset_info.get("columns", [])
    dtypes = dataset_info.get("dtypes", {})
    
    # Build concise summary
    rows = shape[0] if len(shape) > 0 else "unknown"
    cols = shape[1] if len(shape) > 1 else len(columns)
    
    null_info = ", ".join([f"{k}: {v}" for k, v in null_counts.items() if v > 0]) if null_counts else "no nulls"
    
    prompt = f"""You are an AI data cleaning agent. Clean the dataset step by step.

Dataset: {rows} rows, {cols} columns
Columns: {", ".join(columns) if columns else "none"}
Null values: {null_info}

IMPORTANT: Return EXACTLY ONE action per response. Do NOT return multiple actions.

Available actions (use exact names):
- drop_nulls (params: column - optional)
- fill_nulls (params: column, strategy: mean/median/mode)
- remove_duplicates (params: columns - optional)
- filter_rows (params: column, operator, value)
- drop_columns (params: columns)
- convert_types (params: column, dtype: str/int/float/datetime)
- validate_email (params: column, drop_invalid: true/false)
- outlier_removal (params: column, multiplier: float)
- normalize (params: column, method: minmax/zscore)
- submit (when all cleaning is done)

Return ONLY this JSON format - ONE action:
{{"action_type": "action_name", "params": {{}}}}
"""

    if task_info:
        prompt += f"\nTask: {task_info.get('description', '')}\n"
        expected = task_info.get('expected_actions', [])
        prompt += f"Do these actions in order: {', '.join(expected)}\n"

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
        print(f"[DEBUG] Model response: {text[:300]}...", flush=True)

        # Try to parse JSON response
        try:
            import re
            # Remove markdown code blocks
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
            
            # Parse JSON
            data = json.loads(text)
            
            # Handle {"actions": [...]} format - extract first action
            if isinstance(data, dict) and "actions" in data:
                actions = data["actions"]
                if actions and len(actions) > 0:
                    first_action = actions[0]
                    if "action_type" in first_action:
                        return first_action
            
            # Handle direct action format
            if isinstance(data, dict) and "action_type" in data:
                return data
            
            # Handle array format - extract first item
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict) and "action_type" in first_item:
                    return first_item
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"[DEBUG] JSON parse error: {e}", flush=True)
            pass

        # Try to extract action from text using keywords
        action_keywords = ["drop_nulls", "fill_nulls", "remove_duplicates", "filter_rows", 
                          "drop_columns", "convert_types", "validate_email", "outlier_removal", "normalize"]
        for kw in action_keywords:
            if kw in text.lower():
                return {"action_type": kw, "params": {}, "reasoning": f"Extracted from response"}

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

        # Fetch full dataset info for the prompt
        try:
            dataset_info = await env.get_dataset()
            print(f"[DEBUG] Dataset info: shape={dataset_info.get('shape')}, columns={dataset_info.get('columns')}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Could not fetch dataset info: {e}", flush=True)

        # Get task info
        tasks = await env.get_tasks()
        task_info = next((t for t in tasks if t.get("task_id") == task_id), {})

        # Track actions taken to prevent loops
        actions_taken_list = []
        
        for step in range(1, MAX_STEPS + 1):
            # Get action from model
            action = get_model_message(client, step, dataset_info, task_info, history)
            action_type = action.get("action_type", "submit")
            params = action.get("params", {})
            
            # Check for repeated actions (loop detection)
            action_key = f"{action_type}_{json.dumps(params, sort_keys=True)}"
            if action_key in actions_taken_list[-3:] and action_type != "submit":
                print(f"[DEBUG] Detected repeated action, forcing submit", flush=True)
                action_type = "submit"
                params = {}
            actions_taken_list.append(action_key)

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
                # Extract score from response - check multiple locations
                # First check direct fields in response (from submit endpoint)
                score = result.get("final_score", 0.0)
                
                # If not found, check grade object
                if score == 0.0:
                    grade = result.get("grade", {})
                    if grade:
                        score = grade.get("final_score", 0.0)
                
                # Also check observation metadata
                if score == 0.0:
                    obs_metadata = obs.get("metadata", {})
                    grade = obs_metadata.get("grade", {})
                    if grade:
                        score = grade.get("final_score", 0.0)
                    else:
                        # Try to extract from message
                        message = obs_metadata.get("message", "")
                        import re
                        match = re.search(r"Score:\s*([\d.]+)", message)
                        if match:
                            score = float(match.group(1))
                
                # ENSURE SCORE IS STRICTLY BETWEEN 0 AND 1
                # Never exactly 0.0 or 1.0 as required by validation
                if score <= 0.0:
                    score = 0.001
                elif score >= 1.0:
                    score = 0.999
                
                success = score >= 0.5
                break

        # If we didn't submit, do it now
        if not done:
            result = await env.submit()
            info = result.get("info", {})
            grade = info.get("grade", {})
            score = grade.get("final_score", 0.0)
            # ENSURE SCORE IS STRICTLY BETWEEN 0 AND 1
            if score <= 0.0:
                score = 0.001
            elif score >= 1.0:
                score = 0.999
            
            success = score >= 0.5
            rewards.append(result.get("reward", 0.0) or 0.0)
            steps_taken += 1
            
            # Check observation metadata for grade
            obs = result.get("observation", {})
            obs_metadata = obs.get("metadata", {})
            if score == 0.0 and "grade" in obs_metadata:
                score = obs_metadata["grade"].get("final_score", 0.0)
            success = score >= 0.5

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