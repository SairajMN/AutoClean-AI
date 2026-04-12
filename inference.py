#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference.py — AutoClean-AI Inference Script
============================================
Official submission script for OpenEnv Hackathon.

Environment variables (set before running):
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your HuggingFace API key

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py

    # Dry-run without API key (heuristic agent):
    python inference.py --heuristic

    # Run against local dev server:
    python inference.py --env-url http://localhost:7860

Expected baseline scores (heuristic agent, seed=42, 3 episodes x 8 steps):
    easy_001      : ~0.62
    medium_001    : ~0.54
    hard_001      : ~0.41
    overall       : ~0.52
"""

from __future__ import annotations

import os
# Fix Unicode encoding for Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional, Callable

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Structured stdout logging for hackathon evaluation ──────────────────────────
# Required format:
# [START] task=<task_name> env=<benchmark> model=<model_name>
# [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
# [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

BENCHMARK = "openenv-datacleaner"


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] log in required format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] log in required format."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action if too long and handle Unicode
    action_trunc = action[:200].replace("\n", " ") if len(action) > 200 else action.replace("\n", " ")
    # Replace non-ASCII characters to avoid encoding issues
    action_trunc = action_trunc.encode('ascii', 'replace').decode('ascii')
    print(f"[STEP] step={step} action={action_trunc} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] log in required format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Mandatory environment variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "qwen/qwen3-next-80b-a3b-instruct:free")
HF_TOKEN     = os.getenv("OPENROUTER_API_KEY") or os.getenv("HF_TOKEN",     "")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ENV_URL  = os.environ.get(
    "AUTOCLEAN_ENV_URL",
    "https://sairaj2-openenv-datacleaner.hf.space",
)
DEFAULT_EPISODES = 3
DEFAULT_STEPS    = 8
SEED             = 42

TASK_ORDER = [
    ("easy_001",      "beginner"),
    ("medium_001",    "intermediate"),
    ("hard_001",      "advanced"),
]

SYSTEM_PROMPT = """You are an expert data cleaning agent for tabular datasets.

RULES (follow strictly):
1. You are working with a dataset and need to perform data cleaning operations
2. Choose exactly ONE action per step from the allowed actions list
3. Explain your reasoning clearly
4. Always return valid JSON format

ALLOWED ACTIONS:
- drop_nulls
- fill_nulls
- remove_duplicates
- filter_rows
- drop_columns
- convert_types
- validate_email
- outlier_removal
- normalize
- submit
"""

ACTION_PROMPT_TEMPLATE = """DATASET INFORMATION:
{dataset_info}

TASK:
{task_description}

PREVIOUS ACTIONS:
{action_history}

Instructions:
- Select the next best action to clean this dataset
- Provide reasoning for your choice
- Return JSON with these exact keys:
{{
    "action_type": "<action name>",
    "params": {{<parameters for action>}},
    "reasoning": "<short explanation>"
}}"""


# ── Environment client ────────────────────────────────────────────────────────

class EnvClient:
    """Thin HTTP wrapper around the AutoClean REST API."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base       = base_url.rstrip("/")
        self.timeout    = timeout
        self.session    = requests.Session()
        self._session_id: Optional[str] = None

    def _request_with_retry(self, method: str, path: str, body: Dict[str, Any] = None, retries: int = 3, backoff: float = 2.0) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        for attempt in range(retries):
            try:
                if method == "GET":
                    r = self.session.get(url, timeout=self.timeout)
                else:
                    r = self.session.post(url, json=body, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as e:
                if attempt < retries - 1:
                    wait = backoff * (attempt + 1)
                    logger.warning(f"Request to {path} failed ({type(e).__name__}), retrying in {wait:.0f}s... ({attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    raise

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request_with_retry("GET", path)

    def _post(self, path: str, body: Dict[str, Any] = {}) -> Dict[str, Any]:
        return self._request_with_retry("POST", path, body)

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def list_tasks(self) -> Dict[str, Any]:
        return self._get("/tasks")

    def reset(self, difficulty: str, seed: int) -> Dict[str, Any]:
        result = self._post("/reset", {"difficulty": difficulty, "seed": seed})
        self._session_id = result.get("session_id")
        return result

    def step(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "action_type": action_type,
            "params": params,
        }
        if self._session_id:
            body["session_id"] = self._session_id
        return self._post("/step", body)

    def submit(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if self._session_id:
            body["session_id"] = self._session_id
        return self._post("/submit", body)

    def grade(self, task_id: str,
              step_rewards: List[float],
              step_infos:   List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._post("/grader", {
            "task_id":      task_id,
            "step_rewards": step_rewards,
            "step_infos":   step_infos,
        })


# ── Agents ────────────────────────────────────────────────────────────────────

def heuristic_agent(task_id: str, dataset_info: Dict[str, Any], task_description: str, action_history: List[str]) -> Dict[str, Any]:
    """
    Deterministic heuristic baseline — no LLM required.
    Implements standard data cleaning workflows based on task difficulty.
    Used when --heuristic flag is set or no API credentials are available.
    """
    columns = dataset_info.get("columns", [])
    null_counts = dataset_info.get("null_counts", {})
    numeric_columns = [name for name in columns if name in {"age", "salary", "score", "id", "JoiningYear", "ExperienceInCurrentDomain"}]

    def has_taken(action_type: str) -> bool:
        return action_type in action_history

    if task_id == "easy_001":
        if sum(int(v) for v in null_counts.values()) > 0 and not has_taken("drop_nulls"):
            return {"action_type": "drop_nulls", "params": {}}
        if not has_taken("remove_duplicates"):
            return {"action_type": "remove_duplicates", "params": {}}
        return {"action_type": "submit", "params": {}}

    if task_id == "medium_001":
        if sum(int(v) for v in null_counts.values()) > 0 and not has_taken("fill_nulls"):
            target = "age" if "age" in columns else (numeric_columns[0] if numeric_columns else None)
            params = {"column": target, "strategy": "median"} if target else {"strategy": "mode"}
            return {"action_type": "fill_nulls", "params": params}
        if "email" in columns and not has_taken("validate_email"):
            return {"action_type": "validate_email", "params": {"column": "email", "drop_invalid": True}}
        if "salary" in columns and not has_taken("outlier_removal"):
            return {"action_type": "outlier_removal", "params": {"column": "salary", "multiplier": 1.5}}
        return {"action_type": "submit", "params": {}}

    if task_id == "hard_001":
        if sum(int(v) for v in null_counts.values()) > 0 and not has_taken("fill_nulls"):
            target = "salary" if "salary" in columns else (numeric_columns[0] if numeric_columns else None)
            params = {"column": target, "strategy": "median"} if target else {"strategy": "mode"}
            return {"action_type": "fill_nulls", "params": params}
        if not has_taken("remove_duplicates"):
            return {"action_type": "remove_duplicates", "params": {}}
        if "email" in columns and not has_taken("validate_email"):
            return {"action_type": "validate_email", "params": {"column": "email", "drop_invalid": True}}
        if "age" in columns and not has_taken("convert_types"):
            return {"action_type": "convert_types", "params": {"column": "age", "dtype": "int"}}
        if "salary" in columns and not has_taken("outlier_removal"):
            return {"action_type": "outlier_removal", "params": {"column": "salary", "multiplier": 1.5}}
        if "score" in columns and not has_taken("normalize"):
            return {"action_type": "normalize", "params": {"column": "score", "method": "minmax"}}
        return {"action_type": "submit", "params": {}}

    return {"action_type": "submit", "params": {}}


def openai_agent(model: str, base_url: str, api_key: str) -> Callable:
    """
    Returns a callable agent backed by any OpenAI-compatible chat endpoint.
    Uses API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    if not api_key:
        logger.error(
            "HF_TOKEN not set. Export it or use --heuristic for the "
            "no-API baseline.\n"
            "  export HF_TOKEN=hf_..."
        )
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)

    def _call(task_id: str, dataset_info: Dict[str, Any], task_description: str, action_history: List[str]) -> Dict[str, Any]:
        prompt = ACTION_PROMPT_TEMPLATE.format(
            dataset_info=json.dumps(dataset_info, indent=2),
            task_description=task_description,
            action_history=", ".join(action_history) if action_history else "None",
        )

        # Try with JSON response format first, fall back to no format
        for use_json_format in [True, False]:
            try:
                kwargs = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                if use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = client.chat.completions.create(**kwargs)
                raw = (resp.choices[0].message.content or "").strip()

                # Strip markdown code fences if present (```json ... ```)
                import re
                fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
                if fence_match:
                    raw = fence_match.group(1)

                # Try direct JSON parse
                try:
                    parsed = json.loads(raw)
                    action_type = str(parsed.get("action_type", ""))
                    if action_type:
                        return {
                            "action_type": action_type,
                            "params": parsed.get("params", {}),
                        }
                except json.JSONDecodeError:
                    pass

                # Try to extract JSON object from mixed text
                json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', raw, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        return {
                            "action_type": str(parsed.get("action_type", "")),
                            "params": parsed.get("params", {}),
                        }
                    except json.JSONDecodeError:
                        pass

                # If JSON format was tried and failed, fall through to no-format attempt
                if use_json_format:
                    logger.warning("JSON parse failed, trying without response_format")
                    continue

                # Last resort: fall back to heuristic
                return heuristic_agent(task_id, dataset_info, task_description, action_history)

            except Exception as e:
                if use_json_format:
                    error_msg = str(e)
                    if "response_format" in error_msg.lower() or "json_validate_failed" in error_msg:
                        logger.warning(f"JSON format not supported, trying without: {e}")
                        continue
                    else:
                        logger.warning(f"LLM call failed: {e}")
                        return heuristic_agent(task_id, dataset_info, task_description, action_history)
                else:
                    logger.warning(f"LLM call failed: {e}")
                    return heuristic_agent(task_id, dataset_info, task_description, action_history)

        return heuristic_agent(task_id, dataset_info, task_description, action_history)

    return _call


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    env:         EnvClient,
    agent_fn:    Callable,
    task_id:     str,
    difficulty:  str,
    steps:       int,
    seed:        int,
    episode_num: int,
    model_label: str,
    task_info:   Dict[str, Any],
) -> Dict[str, Any]:
    """Run one episode and return rewards + infos for the grader."""
    # Emit START log at beginning of each task
    if episode_num == 0:
        log_start(task=task_id, env=BENCHMARK, model=model_label)

    obs = env.reset(difficulty=difficulty, seed=seed + episode_num)
    step_rewards: List[float]         = []
    step_infos:   List[Dict[str, Any]] = []
    action_history: List[str]         = []

    dataset_info = obs.get("dataset_info", {})

    for step_n in range(steps):
        if obs.get("done", False):
            break

        action = agent_fn(task_id, dataset_info, task_info.get("description", ""), action_history)

        action_type = action.get("action_type", "submit")
        params = action.get("params", {})

        if action_type == "submit":
            obs = env.submit()
        else:
            obs = env.step(action_type, params)

        reward = float(obs.get("reward") or 0.0)
        done = bool(obs.get("done", False))
        step_rewards.append(reward)

        # Extract metrics from observation metadata
        obs_metadata = obs.get("metadata", {})
        step_infos.append({
            "action_type": action_type,
            "reward": reward,
            "done": done,
            "metadata": obs_metadata,
        })

        # Format action for logging
        param_text = ",".join(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in sorted(params.items()))
        action_str = f"{action_type}({param_text})" if param_text else action_type

        # Emit STEP log
        log_step(
            step=step_n + 1,
            action=action_str,
            reward=reward,
            done=done,
            error=None,
        )

        action_history.append(action_type)
        dataset_info = obs.get("dataset_info", dataset_info)

        logger.info(
            f"  [{task_id[:25]}] ep={episode_num+1} step={step_n+1} "
            f"reward={reward:.3f}"
        )

        if done:
            break

    if not done:
        obs = env.submit()
        reward = float(obs.get("reward") or 0.0)
        step_rewards.append(reward)
        log_step(
            step=len(step_rewards),
            action="submit()",
            reward=reward,
            done=True,
            error=None,
        )

    # Calculate score locally (no /grader endpoint)
    episode_score = sum(step_rewards) / max(len(step_rewards), 1)

    return {
        "episode": episode_num + 1,
        "score":   episode_score,
        "rewards": step_rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AutoClean-AI inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-url",   default=DEFAULT_ENV_URL,  help="Environment URL")
    parser.add_argument("--episodes",  type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps",     type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--heuristic", action="store_true",
                        help="Use heuristic agent (no API key needed)")
    parser.add_argument("--output",    default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    # ── Connect to environment ────────────────────────────────────────────────
    env = EnvClient(args.env_url)

    logger.info(f"Connecting to environment: {args.env_url}")
    try:
        h = env.health()
        logger.info(f"  Environment: {h.get('service', 'AutoClean-AI')} v{h.get('version', '1.0.0')} — healthy")
    except Exception as e:
        logger.error(f"Cannot reach environment: {e}")
        sys.exit(1)

    # Verify /tasks endpoint
    try:
        tasks_info = env.list_tasks()
        task_ids   = [t["task_id"] for t in tasks_info.get("tasks", [])]
        task_map   = {t["task_id"]: t for t in tasks_info.get("tasks", [])}
        logger.info(f"  Available tasks: {task_ids}")
    except Exception as e:
        logger.error(f"/tasks endpoint failed: {e}")
        sys.exit(1)

    # ── Select agent ─────────────────────────────────────────────────────────
    if args.heuristic or not HF_TOKEN:
        logger.info("Using heuristic baseline agent (no LLM).")
        agent_fn    = heuristic_agent
        model_label = "heuristic_baseline"
    else:
        logger.info(f"Using LLM agent: {MODEL_NAME} via {API_BASE_URL}")
        agent_fn    = openai_agent(MODEL_NAME, API_BASE_URL, HF_TOKEN)
        model_label = MODEL_NAME

    # ── Run all 3 tasks ───────────────────────────────────────────────────────
    task_results: List[Dict[str, Any]] = []
    all_scores:   List[float]          = []
    all_rewards:  List[float]          = []
    total_steps   = 0
    start_time    = time.time()

    for task_id, difficulty in TASK_ORDER:
        logger.info(f"\n{'='*55}")
        logger.info(f"TASK: {task_id}  (difficulty={difficulty})")
        logger.info(f"{'='*55}")

        episode_scores: List[float] = []
        task_rewards: List[float] = []

        for ep in range(args.episodes):
            ep_result = run_episode(
                env=env,
                agent_fn=agent_fn,
                task_id=task_id,
                difficulty=difficulty,
                steps=args.steps,
                seed=args.seed,
                episode_num=ep,
                model_label=model_label,
                task_info=task_map.get(task_id, {}),
            )
            episode_scores.append(ep_result["score"])
            all_scores.append(ep_result["score"])
            all_rewards.extend(ep_result["rewards"])
            task_rewards.extend(ep_result["rewards"])
            total_steps += len(ep_result["rewards"])

        task_avg = sum(episode_scores) / max(len(episode_scores), 1)
        task_std = (
            (sum((s - task_avg) ** 2 for s in episode_scores) / max(len(episode_scores), 1)) ** 0.5
            if len(episode_scores) > 1 else 0.0
        )

        # Emit [END] log for this task
        success = task_avg >= 0.5  # Consider success if score >= 0.5
        log_end(
            success=success,
            steps=len(task_rewards),
            score=task_avg,
            rewards=task_rewards,
        )

        task_results.append({
            "task_id":        task_id,
            "difficulty":     difficulty,
            "episodes":       args.episodes,
            "episode_scores": [round(s, 4) for s in episode_scores],
            "avg_score":      round(task_avg, 4),
            "std_score":      round(task_std, 4),
        })
        logger.info(f"\n  Task score: {task_avg:.4f} ± {task_std:.4f}")

    elapsed       = time.time() - start_time
    overall_score = sum(all_scores)  / max(len(all_scores),  1)
    avg_reward    = sum(all_rewards) / max(len(all_rewards), 1)

    summary = {
        "model":             model_label,
        "api_base_url":      API_BASE_URL,
        "env_url":           args.env_url,
        "seed":              args.seed,
        "episodes_per_task": args.episodes,
        "steps_per_episode": args.steps,
        "total_steps":       total_steps,
        "elapsed_seconds":   round(elapsed, 1),
        "tasks":             task_results,
        "overall": {
            "score":      round(overall_score, 4),
            "avg_reward": round(avg_reward,    4),
        },
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("INFERENCE RESULTS")
    print("=" * 55)
    print(f"Model      : {model_label}")
    print(f"Seed       : {args.seed}  |  {args.episodes} episodes x {args.steps} steps")
    print(f"Elapsed    : {elapsed:.1f}s")
    print()
    for t in task_results:
        # Use ASCII characters for progress bar
        bar = "#" * round(t["avg_score"] * 20)
        print(
            f"  {t['task_id']:<42} "
            f"{t['avg_score']:.4f} +- {t['std_score']:.4f}  |{bar:<20}|"
        )
    print()
    print(f"  {'OVERALL':<42} {overall_score:.4f}")
    print("=" * 55)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results written to {args.output}")

    return summary


if __name__ == "__main__":
    main()