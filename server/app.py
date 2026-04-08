"""A11yGuard FastAPI application entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.accessibility_env import AccessibilityEnv
from env.models import AccessibilityAction, AccessibilityObservation, EnvironmentState

app = FastAPI(
    title="A11yGuard - Web Accessibility Audit Environment",
    description="OpenEnv environment for evaluating AI agents on WCAG 2.1 AA web accessibility auditing",
    version="1.0.0",
)

_env: Optional[AccessibilityEnv] = None


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    task_name: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: AccessibilityAction


class StepResponse(BaseModel):
    observation: AccessibilityObservation
    reward: float
    done: bool
    info: dict


@app.get("/")
def root():
    return {
        "name": "A11yGuard",
        "description": "Web Accessibility Audit Environment for OpenEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset", response_model=AccessibilityObservation)
def reset(request: ResetRequest = ResetRequest()):
    global _env
    try:
        _env = AccessibilityEnv(
            difficulty=request.difficulty,
            task_name=request.task_name,
            seed=request.seed,
        )
        return _env.reset()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        obs, reward, done, info = _env.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    from env.tasks import get_task_summary
    return get_task_summary()


def start():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    start()
