"""
Orchestrator Agent - Coordinates and aggregates responses from all other agents.
"""
from typing import Dict, Any, List
import asyncio
from datetime import datetime
from agents.base_agent import BaseMCPAgent
from agents.training_agent import TrainingAgent
from agents.diet_agent import DietAgent
from agents.habit_agent import HabitAgent
from agents.qol_agent import QoLAgent
from models.schemas import OrchestratorResponse
import logging

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseMCPAgent):
    """Agent responsible for orchestrating and coordinating all other agents."""
    
    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.training_agent = TrainingAgent()
        self.diet_agent = DietAgent()
        self.habit_agent = HabitAgent()
        self.qol_agent = QoLAgent()
    
    async def start_coaching(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Start comprehensive coaching by orchestrating all agents in parallel."""
        try:
            logger.info(f"Starting coaching session for user {user_id}")
            
            # Get user profile first
            user_profile = await self.get_user_profile(user_id)
            
            # Prepare context for all agents
            agent_context = {
                **context,
                "user_profile": user_profile,
                "coaching_session_id": f"session_{user_id}_{int(datetime.utcnow().timestamp())}"
            }
            
            # Run all agents in parallel
            agent_tasks = [
                self.training_agent.process(user_id, agent_context),
                self.diet_agent.process(user_id, agent_context),
                self.habit_agent.process(user_id, agent_context),
                self.qol_agent.process(user_id, agent_context)
            ]
            
            # Wait for all agents to complete
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process and aggregate results
            aggregated_response = await self._aggregate_agent_responses(
                user_id, agent_results, agent_context
            )
            
            # Log the coaching session
            await self.log_event(user_id, "coaching", {
                "action": "coaching_session_completed",
                "session_id": agent_context["coaching_session_id"],
                "agents_success": [r.get("success", False) for r in agent_results if isinstance(r, dict)],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return aggregated_response
            
        except Exception as e:
            logger.error(f"OrchestratorAgent coaching error: {e}")
            return {
                "agent": self.agent_name,
                "success": False,
                "error": str(e),
                "orchestrator_response": None
            }
    
    async def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process orchestrator-specific logic (delegates to start_coaching)."""
        return await self.start_coaching(user_id, context)
    
    async def _aggregate_agent_responses(
        self, 
        user_id: str, 
        agent_results: List[Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate responses from all agents into a unified response."""
        
        # Initialize response structure
        orchestrator_response = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "training_plan": None,
            "diet_plan": None,
            "habit_suggestions": [],
            "qol_metrics": None,
            "summary": "",
            "next_actions": [],
            "agent_statuses": {}
        }
        
        # Process each agent result
        for i, result in enumerate(agent_results):
            agent_names = ["TrainingAgent", "DietAgent", "HabitAgent", "QoLAgent"]
            agent_name = agent_names[i] if i < len(agent_names) else f"Agent_{i}"
            
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_name} failed with exception: {result}")
                orchestrator_response["agent_statuses"][agent_name] = {
                    "success": False,
                    "error": str(result)
                }
                continue
            
            if not isinstance(result, dict):
                logger.warning(f"Agent {agent_name} returned invalid result type")
                orchestrator_response["agent_statuses"][agent_name] = {
                    "success": False,
                    "error": "Invalid result type"
                }
                continue
            
            # Record agent status
            orchestrator_response["agent_statuses"][agent_name] = {
                "success": result.get("success", False),
                "error": result.get("error")
            }
            
            # Extract agent-specific data
            if agent_name == "TrainingAgent" and result.get("success"):
                orchestrator_response["training_plan"] = result.get("training_plan")
                orchestrator_response["next_actions"].extend(result.get("next_actions", []))
            
            elif agent_name == "DietAgent" and result.get("success"):
                orchestrator_response["diet_plan"] = result.get("diet_plan")
                orchestrator_response["next_actions"].extend(result.get("next_actions", []))
            
            elif agent_name == "HabitAgent" and result.get("success"):
                orchestrator_response["habit_suggestions"] = result.get("habit_suggestions", [])
                orchestrator_response["next_actions"].extend(result.get("next_actions", []))
            
            elif agent_name == "QoLAgent" and result.get("success"):
                orchestrator_response["qol_metrics"] = result.get("qol_metrics")
                orchestrator_response["next_actions"].extend(result.get("next_actions", []))
        
        # Generate comprehensive summary
        orchestrator_response["summary"] = self._generate_summary(orchestrator_response)
        
        # Prioritize and deduplicate next actions
        orchestrator_response["next_actions"] = self._prioritize_actions(
            orchestrator_response["next_actions"]
        )
        
        return {
            "agent": self.agent_name,
            "success": True,
            "orchestrator_response": orchestrator_response,
            "coaching_session_id": context.get("coaching_session_id"),
            "recommendations": self._generate_orchestrator_recommendations(orchestrator_response)
        }
    
    def _generate_summary(self, response: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of the coaching session."""
        summary_parts = []
        
        # Check what was successfully generated
        successful_agents = [
            name for name, status in response["agent_statuses"].items() 
            if status.get("success", False)
        ]
        
        if not successful_agents:
            return "Coaching session encountered issues. Please try again or contact support."
        
        summary_parts.append(f"Coaching session completed successfully with {len(successful_agents)} agents.")
        
        # Training summary
        if response["training_plan"]:
            plan = response["training_plan"]
            summary_parts.append(
                f"Training: {plan.get('duration_weeks', 8)}-week program with "
                f"{plan.get('frequency_per_week', 3)} workouts per week."
            )
        
        # Diet summary
        if response["diet_plan"]:
            plan = response["diet_plan"]
            summary_parts.append(
                f"Nutrition: {plan.get('daily_calories', 2000)} calorie daily plan with "
                f"{len(plan.get('meals', []))} structured meals."
            )
        
        # Habit summary
        if response["habit_suggestions"]:
            habits = response["habit_suggestions"]
            summary_parts.append(
                f"Habits: {len(habits)} personalized habit suggestions across "
                f"{len(set(h.get('category', '') for h in habits))} categories."
            )
        
        # QoL summary
        if response["qol_metrics"]:
            qol = response["qol_metrics"]
            summary_parts.append(
                f"Quality of Life: Overall score of {qol.get('overall_score', 0)}/10 with "
                f"targeted improvement recommendations."
            )
        
        return " ".join(summary_parts)
    
    def _prioritize_actions(self, actions: List[str]) -> List[str]:
        """Prioritize and deduplicate next actions."""
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        # Define priority keywords
        priority_keywords = {
            "start": 10,
            "begin": 10,
            "plan": 8,
            "track": 7,
            "focus": 6,
            "implement": 5,
            "adjust": 3,
            "review": 2
        }
        
        # Sort by priority
        def get_priority(action):
            action_lower = action.lower()
            for keyword, priority in priority_keywords.items():
                if keyword in action_lower:
                    return priority
            return 1  # Default priority
        
        sorted_actions = sorted(unique_actions, key=get_priority, reverse=True)
        
        # Limit to top 8 actions
        return sorted_actions[:8]
    
    def _generate_orchestrator_recommendations(self, response: Dict[str, Any]) -> List[str]:
        """Generate high-level orchestrator recommendations."""
        recommendations = []
        
        # Check overall success rate
        successful_count = sum(
            1 for status in response["agent_statuses"].values() 
            if status.get("success", False)
        )
        total_agents = len(response["agent_statuses"])
        
        if successful_count == total_agents:
            recommendations.append("All systems are ready! Start with your personalized plans.")
        elif successful_count >= total_agents * 0.75:
            recommendations.append("Most plans are ready. Focus on available recommendations first.")
        else:
            recommendations.append("Some systems need attention. Contact support if issues persist.")
        
        # QoL-based recommendations
        qol_metrics = response.get("qol_metrics")
        if qol_metrics:
            overall_score = qol_metrics.get("overall_score", 5)
            if overall_score < 5:
                recommendations.append("Focus on foundational wellness before intensive programs.")
            elif overall_score > 7:
                recommendations.append("You're in great shape! Consider advanced challenges.")
        
        # Integration recommendations
        if response["training_plan"] and response["diet_plan"]:
            recommendations.append("Coordinate your training and nutrition for optimal results.")
        
        if response["habit_suggestions"]:
            recommendations.append("Start with 1-2 habits to build momentum before adding more.")
        
        # General recommendations
        recommendations.extend([
            "Review your progress weekly and adjust plans as needed",
            "Stay consistent with small daily actions",
            "Celebrate your progress along the way"
        ])
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    async def get_coaching_status(self, user_id: str) -> Dict[str, Any]:
        """Get current coaching status for a user."""
        try:
            # Get recent coaching events
            coaching_history = await self.fetch_historical_data(user_id, "coaching", 7)
            
            # Get user profile
            user_profile = await self.get_user_profile(user_id)
            
            # Calculate progress metrics
            progress_metrics = await self._calculate_progress_metrics(user_id)
            
            return {
                "user_id": user_id,
                "last_coaching_session": coaching_history.get("data", [])[-1] if coaching_history.get("data") else None,
                "active_goals": len(user_profile.get("goals", [])),
                "progress_metrics": progress_metrics,
                "next_session_recommended": self._recommend_next_session(coaching_history),
                "status": "active" if coaching_history.get("data") else "inactive"
            }
            
        except Exception as e:
            logger.error(f"Error getting coaching status: {e}")
            return {"error": str(e)}
    
    async def _calculate_progress_metrics(self, user_id: str) -> Dict[str, Any]:
        """Calculate user progress metrics across all areas."""
        # This would typically involve more complex calculations
        # For now, return basic metrics
        return {
            "goals_completed": 0,
            "habits_established": 0,
            "workouts_completed": 0,
            "nutrition_compliance": 0.0,
            "overall_progress": 0.0
        }
    
    def _recommend_next_session(self, coaching_history: Dict) -> str:
        """Recommend when the next coaching session should occur."""
        data = coaching_history.get("data", [])
        
        if not data:
            return "Start your first coaching session now"
        
        # Simple recommendation based on last session
        return "Weekly check-in recommended"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get OrchestratorAgent information."""
        return {
            "name": self.agent_name,
            "description": "Coordinates and aggregates responses from all health and wellness agents",
            "capabilities": [
                "Multi-agent coordination",
                "Parallel processing",
                "Response aggregation",
                "Comprehensive coaching sessions",
                "Progress tracking",
                "Recommendation prioritization"
            ],
            "managed_agents": [
                "TrainingAgent",
                "DietAgent", 
                "HabitAgent",
                "QoLAgent"
            ],
            "endpoints": [
                "/api/orchestrator/start-coaching",
                "/api/orchestrator/status",
                "/api/orchestrator/progress"
            ]
        }
