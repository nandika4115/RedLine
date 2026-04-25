"""
RedLine — FastAPI HTTP server.
Exposes /reset, /step, /state endpoints per OpenEnv spec.
Run: uvicorn RedLine.app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import ClinicalAction, ClinicalObservation, EpisodeState
from .server import ClinicalTrialEnv

app = FastAPI(title="RedLine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One env per server process (fine for hackathon; use session IDs for prod)
_env = ClinicalTrialEnv(max_steps=50)


class StepResponse(BaseModel):
    observation: ClinicalObservation
    reward: float
    done: bool


@app.post("/reset", response_model=ClinicalObservation)
async def reset():
    return _env.reset()


@app.post("/step", response_model=StepResponse)
async def step(action: ClinicalAction):
    try:
        obs, reward, done = _env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done)


@app.get("/state", response_model=EpisodeState)
async def state():
    return _env.state()


@app.get("/health")
async def health():
    return {"status": "ok"}
