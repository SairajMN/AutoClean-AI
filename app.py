"""
OpenEnv Data Cleaning Environment - FastAPI Server
Thin wrapper over OpenEnv-compliant DataCleaningEnv.
Production-ready server for Hugging Face Spaces deployment.
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import OpenEnv-compliant environment
from env.datacleaner_env import DataCleaningEnv
from env.models import Action, Observation, Reward
from env.tasks import get_tasks, get_all_task_configs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("openenv-datacleaner")

# Initialize FastAPI app
app = FastAPI(
    title="OpenEnv Data Cleaner",
    description="OpenEnv-compliant AI-powered data cleaning environment for Hugging Face Spaces",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Request/Response Models
# ============================================================

class ResetRequest(BaseModel):
    task_id: str = "easy_001"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action_type: str
    params: Dict[str, Any] = {}

class TaskInfo(BaseModel):
    task_id: str
    difficulty: str
    description: str
    expected_actions: List[str]

# ============================================================
# Environment Instance
# ============================================================

# Global OpenEnv environment instance
openenv_env: Optional[DataCleaningEnv] = None
start_time: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize OpenEnv environment on startup."""
    global openenv_env, start_time
    
    logger.info("=" * 60)
    logger.info("OpenEnv Data Cleaner - Starting Up")
    logger.info("=" * 60)
    
    # Create OpenEnv-compliant environment
    openenv_env = DataCleaningEnv()
    start_time = time.time()
    
    logger.info("OpenEnv environment initialized")
    logger.info("Startup complete. Environment ready.")


# ============================================================
# Core OpenEnv Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK with environment status.
    """
    uptime = time.time() - start_time if start_time else 0
    
    return {
        "status": "healthy",
        "session_id": openenv_env.get_session_id() if openenv_env else None,
        "task_id": openenv_env.get_task_id() if openenv_env else None,
        "uptime_seconds": round(uptime, 2),
        "step_count": openenv_env.state().step_count if openenv_env else 0,
        "done": openenv_env.is_done() if openenv_env else False,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/reset")
async def reset_environment(request: ResetRequest):
    """
    Reset the OpenEnv environment and initialize a new task.
    Calls env.reset(task_id, session_id).
    """
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    try:
        logger.info(f"Resetting environment: task_id={request.task_id}")
        
        # Call OpenEnv reset
        observation = openenv_env.reset(
            task_id=request.task_id,
            session_id=request.session_id
        )
        
        return {
            "status": "reset_complete",
            "observation": observation.model_dump(),
            "session_id": openenv_env.get_session_id(),
            "task_id": openenv_env.get_task_id()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def execute_step(request: StepRequest):
    """
    Execute an action in the OpenEnv environment.
    Calls env.step(action) and returns (observation, reward, done, info).
    """
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    try:
        # Create Action model
        action = Action(
            action_type=request.action_type,
            params=request.params,
            task_id=openenv_env.get_task_id()
        )
        
        logger.info(f"Executing step: {request.action_type}")
        
        # Call OpenEnv step
        observation, reward, done, info = openenv_env.step(action)
        
        return {
            "status": "success",
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
        
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/tasks")
async def get_available_tasks():
    """
    Get available OpenEnv tasks.
    Returns task list with configurations.
    """
    task_ids = get_tasks()
    configs = get_all_task_configs()
    
    tasks = []
    for task_id in task_ids:
        config = configs[task_id]
        tasks.append({
            "task_id": config.task_id,
            "difficulty": config.difficulty,
            "description": config.description,
            "expected_actions": config.expected_actions,
            "grading_criteria": config.grading_criteria
        })
    
    return {
        "tasks": tasks,
        "count": len(tasks)
    }


# ============================================================
# State & Info Endpoints
# ============================================================

@app.get("/state")
async def get_state():
    """Get current OpenEnv environment state."""
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    state = openenv_env.state()
    return state.model_dump()


@app.get("/dataset")
async def get_dataset_info():
    """Get current dataset information."""
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    dataset = openenv_env.get_dataset()
    if dataset is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Reset environment first.")
    
    return {
        "session_id": openenv_env.get_session_id(),
        "task_id": openenv_env.get_task_id(),
        "shape": list(dataset.shape),
        "columns": dataset.columns.tolist(),
        "null_counts": {
            col: int(dataset[col].isnull().sum())
            for col in dataset.columns
        },
        "dtypes": {
            col: str(dtype)
            for col, dtype in dataset.dtypes.items()
        },
        "step_count": openenv_env.state().step_count
    }


@app.get("/history")
async def get_history():
    """Get action execution history."""
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    return {
        "session_id": openenv_env.get_session_id(),
        "task_id": openenv_env.get_task_id(),
        "step_count": openenv_env.state().step_count,
        "done": openenv_env.is_done(),
        "history": openenv_env.get_action_history()
    }


@app.post("/revert")
async def revert_last_action():
    """Revert the last executed action."""
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    # Create revert action
    action = Action(
        action_type="revert",
        params={},
        task_id=openenv_env.get_task_id()
    )
    
    observation, reward, done, info = openenv_env.step(action)
    
    return {
        "status": "reverted",
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "info": info
    }


@app.post("/submit")
async def submit_solution():
    """Submit the current solution for grading."""
    if openenv_env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    # Create submit action
    action = Action(
        action_type="submit",
        params={},
        task_id=openenv_env.get_task_id()
    )
    
    observation, reward, done, info = openenv_env.step(action)
    
    return {
        "status": "submitted",
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting OpenEnv Data Cleaner on port {port}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )