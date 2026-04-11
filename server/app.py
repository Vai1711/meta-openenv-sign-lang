#!/usr/bin/env python3
"""
Main entry point for OpenEnv server on Hugging Face
Starts the Sign Language Interpreter environment on port 7860
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Add both the server folder AND the root folder to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure these imports match your file structure
try:
    from env import SignInterpreterEnv, SignAction, ActionType
except ImportError:
    # Fallback if running from within the server directory
    from ..env import SignInterpreterEnv, SignAction, ActionType

app = FastAPI(title="Sign Language Interpreter Environment", version="1.0.0")

# Global environment instance
env = SignInterpreterEnv(max_steps=10, seed=42)

class ResetRequest(BaseModel):
    """Request model for environment reset"""
    task_id: Optional[int] = None

class ActionRequest(BaseModel):
    """Request model for actions"""
    action_type: str
    query_sign: Optional[str] = None
    query_context: Optional[str] = None
    translation: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"message": "Sign Language Interpreter Environment", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "environment": "SignLanguageInterpreter"}

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment. 
    Crucial: Grader passes task_id here to select difficulty.
    """
    try:
        task_id = request.task_id if request else None
        observation = env.reset(task_id=task_id)
        
        return {
            "observation": observation.model_dump(),
            "success": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )

@app.post("/step")
async def step(request: ActionRequest):
    """Take a step in the environment"""
    try:
        action_type_map = {
            "query_dict": ActionType.QUERY_DICT,
            "query_context": ActionType.QUERY_CONTEXT,
            "submit_translation": ActionType.SUBMIT_TRANSLATION
        }
        
        if request.action_type not in action_type_map:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid action_type: {request.action_type}", "success": False}
            )
        
        action = SignAction(
            action_type=action_type_map[request.action_type],
            query_sign=request.query_sign,
            query_context=request.query_context,
            translation=request.translation
        )
        
        if not action.validate_action():
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid action parameters", "success": False}
            )
        
        observation, reward, done, info = env.step(action)
        
        return {
            "observation": observation.model_dump(),
            "reward": float(reward),
            "done": bool(done),
            "info": info,
            "success": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )

@app.get("/state")
async def state():
    """Get current environment state"""
    try:
        # Check if env.state() exists, otherwise return general info
        state_data = env.state() if hasattr(env, 'state') else {"status": "active"}
        return {"state": state_data, "success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/info")
async def info():
    """Returns environment metadata to the grader"""
    return {
        "name": "Sign Language Interpreter",
        "version": "1.0.0",
        "action_types": ["query_dict", "query_context", "submit_translation"],
        "difficulty_levels": ["easy", "medium", "hard"],
        "success": True
    }

# CRITICAL FIX: The grader needs a callable main() function 
# for multi-mode deployment validation.
def main():
    """Callable entry point for the OpenEnv validator"""
    port = int(os.getenv("PORT", 7860))
    # We use "server.app:app" because uvicorn needs the module path
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
