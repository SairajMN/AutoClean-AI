---
title: AutoClean-Ai
emoji: 🧹
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
  - data-preprocessing
  - llm-training
  - benchmark
  - ai-safety
  - data-quality
  - mlops
---

# 🧹 AutoClean-Ai

> **Production-grade OpenEnv RL environment for training AI models to clean tabular data automatically.**

**Server Version:** v1.0.0

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](#-quick-start)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Realistic%20Generated-orange)](#-datasets)

---

## 💡 The Problem

80% of data scientist time is spent cleaning data. Bad data causes 60% of ML project failures. AutoClean-Ai was built to train AI agents that can automatically detect and fix common data quality issues in tabular datasets.

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/SairajMN/WorkflowOps.git
cd WorkflowOps
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
curl http://localhost:7860/health
```

### Raw HTTP

```python
import requests

BASE = "http://localhost:7860"

# 1. Start episode
obs = requests.post(f"{BASE}/reset", json={"difficulty": "beginner"}).json()
print(obs["dataset_preview"], obs["column_info"])

# 2. Submit cleaning action
result = requests.post(f"{BASE}/step", json={
    "action_type": "fix_missing_values",
    "column_index": 2,
    "confidence": 0.92,
    "reasoning": "Mean imputation for numerical column",
    "session_id": obs.get("session_id"),
}).json()
print(f"Reward: {result['reward']}, Cleaned: {result['rows_cleaned']}")

# 3. Score the episode
grade = requests.post(f"{BASE}/grader", json={
    "task_id": "task_1_basic_cleaning",
    "step_rewards": [result['reward']],
    "step_infos": [result],
}).json()
print(f"Episode score: {grade['score']}")
```

### Validate OpenEnv Compliance

```bash
# Local structure check
openenv validate

# Runtime check against live server
openenv validate --url http://localhost:7860 --verbose
```

---
```bash
python3 inference.py

2026-04-12 22:19:47,173 [INFO] Connecting to environment: https://sairaj2-openenv-datacleaner.hf.space
2026-04-12 22:19:49,338 [INFO]   Environment: AutoClean-AI v1.0.0 — healthy
2026-04-12 22:19:49,711 [INFO]   Available tasks: ['easy_001', 'medium_001', 'hard_001', 'employee_demo']
2026-04-12 22:19:49,711 [INFO] Using LLM agent: qwen/qwen3-next-80b-a3b-instruct:free via https://openrouter.ai/api/v1
2026-04-12 22:19:50,044 [INFO] 
=======================================================
2026-04-12 22:19:50,044 [INFO] TASK: easy_001  (difficulty=beginner)
2026-04-12 22:19:50,044 [INFO] =======================================================
[START] task=easy_001 env=openenv-datacleaner model=qwen/qwen3-next-80b-a3b-instruct:free
2026-04-12 22:19:52,471 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:19:52,472 [INFO] Retrying request to /chat/completions in 0.464138 seconds
2026-04-12 22:19:53,580 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:19:53,580 [INFO] Retrying request to /chat/completions in 0.815704 seconds
2026-04-12 22:19:55,038 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:19:55,041 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'qwen/qwen3-next-80b-a3b-instruct:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Venice', 'is_byok': False}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=remove_duplicates reward=0.50 done=false error=null
2026-04-12 22:19:55,383 [INFO]   [easy_001] ep=1 step=1 reward=0.500
2026-04-12 22:19:55,965 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:19:55,967 [INFO] Retrying request to /chat/completions in 0.458206 seconds
2026-04-12 22:19:57,083 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 402 Payment Required"
2026-04-12 22:19:57,084 [WARNING] LLM call failed: Error code: 402 - {'error': {'message': 'Provider returned error', 'code': 402, 'metadata': {'raw': '{"error":"API key USD spend limit exceeded. Your account may still have USD balance, but this API key has reached its configured USD spending limit."}', 'provider_name': 'Venice', 'is_byok': False}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=2 action=submit reward=1.00 done=true error=null
2026-04-12 22:19:57,485 [INFO]   [easy_001] ep=1 step=2 reward=1.000
2026-04-12 22:19:58,443 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:19:58,445 [INFO] Retrying request to /chat/completions in 0.475367 seconds
2026-04-12 22:19:59,627 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:19:59,628 [INFO] Retrying request to /chat/completions in 0.844512 seconds
2026-04-12 22:20:01,065 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:01,067 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'qwen/qwen3-next-80b-a3b-instruct:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Venice', 'is_byok': False}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=remove_duplicates reward=0.50 done=false error=null
2026-04-12 22:20:01,372 [INFO]   [easy_001] ep=2 step=1 reward=0.500
2026-04-12 22:20:01,969 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:01,971 [INFO] Retrying request to /chat/completions in 0.387579 seconds
2026-04-12 22:20:03,191 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:03,193 [INFO] Retrying request to /chat/completions in 0.930048 seconds
2026-04-12 22:20:04,715 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:04,717 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'qwen/qwen3-next-80b-a3b-instruct:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Venice', 'is_byok': False}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=2 action=submit reward=1.00 done=true error=null
2026-04-12 22:20:05,054 [INFO]   [easy_001] ep=2 step=2 reward=1.000
2026-04-12 22:20:06,558 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:06,560 [INFO] Retrying request to /chat/completions in 0.377761 seconds
2026-04-12 22:20:08,138 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:08,139 [INFO] Retrying request to /chat/completions in 0.790773 seconds
2026-04-12 22:20:09,531 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:09,533 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Provider returned error', 'code': 429, 'metadata': {'raw': 'qwen/qwen3-next-80b-a3b-instruct:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations', 'provider_name': 'Venice', 'is_byok': False}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=remove_duplicates reward=0.50 done=false error=null
2026-04-12 22:20:09,877 [INFO]   [easy_001] ep=3 step=1 reward=0.500
2026-04-12 22:20:10,478 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:10,480 [INFO] Retrying request to /chat/completions in 0.432287 seconds
2026-04-12 22:20:11,245 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:11,247 [INFO] Retrying request to /chat/completions in 0.841678 seconds
2026-04-12 22:20:12,445 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:12,447 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=2 action=submit reward=1.00 done=true error=null
2026-04-12 22:20:12,771 [INFO]   [easy_001] ep=3 step=2 reward=1.000
[END] success=true steps=6 score=0.750 rewards=0.50,1.00,0.50,1.00,0.50,1.00
2026-04-12 22:20:12,771 [INFO] 
  Task score: 0.7500 ± 0.0000
2026-04-12 22:20:12,771 [INFO] 
=======================================================
2026-04-12 22:20:12,771 [INFO] TASK: medium_001  (difficulty=intermediate)
2026-04-12 22:20:12,771 [INFO] =======================================================
[START] task=medium_001 env=openenv-datacleaner model=qwen/qwen3-next-80b-a3b-instruct:free
2026-04-12 22:20:13,504 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:13,504 [INFO] Retrying request to /chat/completions in 0.469513 seconds
2026-04-12 22:20:14,323 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:14,323 [INFO] Retrying request to /chat/completions in 0.933486 seconds
2026-04-12 22:20:16,371 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:16,371 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=submit reward=0.50 done=true error=null
2026-04-12 22:20:16,811 [INFO]   [medium_001] ep=1 step=1 reward=0.500
2026-04-12 22:20:17,561 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:17,562 [INFO] Retrying request to /chat/completions in 0.445498 seconds
2026-04-12 22:20:18,419 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:18,421 [INFO] Retrying request to /chat/completions in 0.807103 seconds
2026-04-12 22:20:19,640 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:19,641 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=submit reward=0.50 done=true error=null
2026-04-12 22:20:19,980 [INFO]   [medium_001] ep=2 step=1 reward=0.500
2026-04-12 22:20:20,626 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:20,627 [INFO] Retrying request to /chat/completions in 0.397460 seconds
2026-04-12 22:20:21,491 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:21,493 [INFO] Retrying request to /chat/completions in 0.964606 seconds
2026-04-12 22:20:22,821 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:22,823 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=submit reward=0.50 done=true error=null
2026-04-12 22:20:23,198 [INFO]   [medium_001] ep=3 step=1 reward=0.500
[END] success=true steps=3 score=0.500 rewards=0.50,0.50,0.50
2026-04-12 22:20:23,199 [INFO] 
  Task score: 0.5000 ± 0.0000
2026-04-12 22:20:23,199 [INFO] 
=======================================================
2026-04-12 22:20:23,199 [INFO] TASK: hard_001  (difficulty=advanced)
2026-04-12 22:20:23,199 [INFO] =======================================================
[START] task=hard_001 env=openenv-datacleaner model=qwen/qwen3-next-80b-a3b-instruct:free
2026-04-12 22:20:24,051 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:24,052 [INFO] Retrying request to /chat/completions in 0.472201 seconds
2026-04-12 22:20:25,173 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:25,174 [INFO] Retrying request to /chat/completions in 0.768212 seconds
2026-04-12 22:20:26,285 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:26,286 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=remove_duplicates reward=0.50 done=false error=null
2026-04-12 22:20:26,614 [INFO]   [hard_001] ep=1 step=1 reward=0.500
2026-04-12 22:20:27,026 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:27,026 [INFO] Retrying request to /chat/completions in 0.446455 seconds
2026-04-12 22:20:28,422 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:28,424 [INFO] Retrying request to /chat/completions in 0.765570 seconds
2026-04-12 22:20:29,526 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:29,527 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=2 action=submit reward=1.00 done=true error=null
2026-04-12 22:20:29,927 [INFO]   [hard_001] ep=1 step=2 reward=1.000
2026-04-12 22:20:30,587 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:30,589 [INFO] Retrying request to /chat/completions in 0.408676 seconds
2026-04-12 22:20:31,424 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:31,426 [INFO] Retrying request to /chat/completions in 0.778604 seconds
2026-04-12 22:20:32,608 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:32,611 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=remove_duplicates reward=0.50 done=false error=null
2026-04-12 22:20:33,065 [INFO]   [hard_001] ep=2 step=1 reward=0.500
2026-04-12 22:20:33,472 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:33,473 [INFO] Retrying request to /chat/completions in 0.458515 seconds
2026-04-12 22:20:34,394 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:34,395 [INFO] Retrying request to /chat/completions in 0.825773 seconds
2026-04-12 22:20:35,545 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:35,547 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=2 action=submit reward=1.00 done=true error=null
2026-04-12 22:20:35,874 [INFO]   [hard_001] ep=2 step=2 reward=1.000
2026-04-12 22:20:36,572 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:36,573 [INFO] Retrying request to /chat/completions in 0.417865 seconds
2026-04-12 22:20:37,307 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:37,309 [INFO] Retrying request to /chat/completions in 0.985335 seconds
2026-04-12 22:20:38,616 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:38,618 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=1 action=remove_duplicates reward=0.50 done=false error=null
2026-04-12 22:20:38,959 [INFO]   [hard_001] ep=3 step=1 reward=0.500
2026-04-12 22:20:39,310 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:39,311 [INFO] Retrying request to /chat/completions in 0.375729 seconds
2026-04-12 22:20:40,045 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:40,046 [INFO] Retrying request to /chat/completions in 0.926493 seconds
2026-04-12 22:20:41,322 [INFO] HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-04-12 22:20:41,325 [WARNING] LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit exceeded: limit_rpm/qwen/qwen3-next-80b-a3b-instruct-2509/94248808-ba97-4e3c-be60-1cb0928db51d. High demand for qwen/qwen3-next-80b-a3b-instruct:free on OpenRouter - limited to 8 requests per minute. Please retry shortly.', 'code': 429, 'metadata': {'headers': {'X-RateLimit-Limit': '8', 'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1776012660000'}, 'provider_name': None}}, 'user_id': 'user_36ZHxohbiGTyLfq9vP3Sf6ojZMM'}
[STEP] step=2 action=submit reward=1.00 done=true error=null
2026-04-12 22:20:41,690 [INFO]   [hard_001] ep=3 step=2 reward=1.000
[END] success=true steps=6 score=0.750 rewards=0.50,1.00,0.50,1.00,0.50,1.00
2026-04-12 22:20:41,690 [INFO] 
  Task score: 0.7500 ± 0.0000

=======================================================
INFERENCE RESULTS
=======================================================
Model      : qwen/qwen3-next-80b-a3b-instruct:free
Seed       : 42  |  3 episodes x 8 steps
Elapsed    : 51.6s

  easy_001                                   0.7500 +- 0.0000  |###############     |
  medium_001                                 0.5000 +- 0.0000  |##########          |
  hard_001                                   0.7500 +- 0.0000  |###############     |

  OVERALL                                    0.6667
=======================================================
```

---

---

## 🎯 Tasks

3 progressive difficulty tasks:

| # | task_id | Difficulty | Description | Expected Agent Score |
|---|---------|-----------|-------------|-------------------|
| 1 | `task_1_basic_cleaning` | 🟢 Beginner | Fix missing values, standardize formats | 0.70–0.85 |
| 2 | `task_2_advanced_cleaning` | 🟡 Intermediate | Handle outliers, correct data types, deduplication | 0.55–0.70 |
| 3 | `task_3_full_pipeline` | 🔴 Advanced | Complete end-to-end data cleaning pipeline | 0.40–0.60 |

---

## 🎮 Environment Workflow

The agent receives a **tabular dataset** with known quality issues. It must select the appropriate cleaning operation, apply it correctly, and justify its choice.

### Action Space

```json
{
    "action_type":      "fix_missing_values | remove_outliers | standardize | deduplicate | correct_types | fill_dates",
    "column_index":     3,
    "confidence":       0.85,
    "reasoning":        "string explaining the choice",
    "session_id":       "session id from reset"
}
```

### Observation Space

```json
{
    "dataset_preview":   "First 5 rows of data",
    "column_info":       "Column names, types, missing stats",
    "reward":            0.75,
    "feedback":          "Detailed human-readable feedback",
    "rows_cleaned":      12,
    "issues_remaining":  3,
    "done":              false,
    "session_id":        "ses_a1b2c3d4"
}
```

---

## 📊 Reward System (7 Components)

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.35 | Operation actually fixed the issue |
| Appropriate action | 0.25 | Right operation selected for the problem |
| Confidence calibration | 0.15 | Confidence matches actual correctness |
| No side effects | 0.15 | Cleaning didn't break other columns |
| Efficiency | 0.10 | Minimum steps to clean dataset |

---

## 📈 Metrics

✅ Data Quality Score
✅ Completeness Ratio
✅ Uniqueness Ratio
✅ Type Consistency
✅ Cleaning Efficiency
✅ Action Appropriateness

---

## 📋 Supported Data Cleaning Operations

| Operation | Description |
|-----------|-------------|
| `fix_missing_values` | Mean/median/mode imputation |
| `remove_outliers` | IQR / Z-score outlier removal |
| `standardize` | Normalize numerical columns |
| `deduplicate` | Remove duplicate rows |
| `correct_types` | Fix incorrect data types |
| `fill_dates` | Standardize date formats |
| `handle_categories` | Encode categorical columns |
| `remove_duplicates` | Drop identical rows |
| `trim_strings` | Clean whitespace from text columns |
| `correct_values` | Fix known invalid values |

---

## 📀 API Endpoints

### OpenEnv Required

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tasks` | List all 3 tasks + action schema |
| `POST` | `/grader` | Score a completed episode (0.0–1.0) |
| `POST` | `/baseline` | Run built-in heuristic baseline |
| `GET` | `/metadata` | Environment name, version, description |
| `GET` | `/schema` | Action, observation, and state JSON schemas |
| `GET` | `/health` | Health check |

### Environment

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Submit cleaning action |
| `GET` | `/state` | Get current episode state |

---

## 💻 Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Validate OpenEnv compliance
openenv validate --url http://localhost:7860 --verbose
```

---

## 🔗 Links

| | |
|---|---|
| 📦 GitHub | https://github.com/SairajMN/AutoClean-AI |
| 📖 Interactive API Docs | http://localhost:7860/redoc |
| 🔧 OpenEnv Framework | https://github.com/meta-pytorch/OpenEnv |

---

*Built for Data Cleaning AI Agents · MIT License*