"""
Orchestrator API routes for the Health & Quality of Life MCP App.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any, Optional
from agents.orchestrator_agent import OrchestratorAgent
from core.auth import get_current_user
from models.schemas import OrchestratorResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize orchestrator agent
orchestrator = OrchestratorAgent()


@router.post("/start-coaching")
async def start_coaching(
    context: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Start comprehensive coaching session with all agents."""
    try:
        user_id = current_user["user_id"]
        
        # Start coaching session
        result = await orchestrator.start_coaching(user_id, context)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Coaching session failed: {result.get('error', 'Unknown error')}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start coaching error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start coaching session"
        )


@router.get("/status")
async def get_coaching_status(current_user: dict = Depends(get_current_user)):
    """Get current coaching status for the user."""
    try:
        user_id = current_user["user_id"]
        status_info = await orchestrator.get_coaching_status(user_id)
        
        return status_info
        
    except Exception as e:
        logger.error(f"Get coaching status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get coaching status"
        )


@router.get("/agents/info")
async def get_agents_info():
    """Get information about all managed agents."""
    try:
        agents_info = {
            "orchestrator": orchestrator.get_agent_info(),
            "training": orchestrator.training_agent.get_agent_info(),
            "diet": orchestrator.diet_agent.get_agent_info(),
            "habit": orchestrator.habit_agent.get_agent_info(),
            "qol": orchestrator.qol_agent.get_agent_info()
        }
        
        return {
            "total_agents": len(agents_info),
            "agents": agents_info
        }
        
    except Exception as e:
        logger.error(f"Get agents info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get agents information"
        )


@router.post("/agents/{agent_name}/process")
async def process_individual_agent(
    agent_name: str,
    context: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Process a specific agent individually."""
    try:
        user_id = current_user["user_id"]
        
        # Get the specific agent
        agent_map = {
            "training": orchestrator.training_agent,
            "diet": orchestrator.diet_agent,
            "habit": orchestrator.habit_agent,
            "qol": orchestrator.qol_agent
        }
        
        if agent_name not in agent_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found"
            )
        
        agent = agent_map[agent_name]
        result = await agent.process(user_id, context)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process agent {agent_name} error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process {agent_name} agent"
        )


@router.get("/progress")
async def get_user_progress(current_user: dict = Depends(get_current_user)):
    """Get comprehensive user progress across all areas."""
    try:
        user_id = current_user["user_id"]
        
        # Get progress from orchestrator
        progress = await orchestrator._calculate_progress_metrics(user_id)
        
        # Get recent coaching history
        coaching_history = await orchestrator.fetch_historical_data(user_id, "coaching", 30)
        
        return {
            "user_id": user_id,
            "progress_metrics": progress,
            "coaching_sessions": len(coaching_history.get("data", [])),
            "last_updated": coaching_history.get("data", [])[-1].get("timestamp") if coaching_history.get("data") else None
        }
        
    except Exception as e:
        logger.error(f"Get user progress error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user progress"
        )
