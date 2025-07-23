"""
Model Context Protocol (MCP) Server implementation for Health & Quality of Life App.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

from core.database import get_supabase
from models.schemas import MCPTool, MCPResource, MCPPrompt, MCPInvokeRequest, MCPInvokeResponse

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server implementation with tools, resources, and prompts."""
    
    def __init__(self):
        self.tools = self._initialize_tools()
        self.resources = self._initialize_resources()
        self.prompts = self._initialize_prompts()
    
    def _initialize_tools(self) -> Dict[str, MCPTool]:
        """Initialize MCP tools."""
        return {
            "get_user_profile": MCPTool(
                name="get_user_profile",
                description="Retrieve user profile information including goals and preferences",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Unique identifier for the user"
                        }
                    },
                    "required": ["user_id"]
                }
            ),
            "log_event": MCPTool(
                name="log_event",
                description="Log user events and activities for tracking and analysis",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Unique identifier for the user"
                        },
                        "event_type": {
                            "type": "string",
                            "enum": ["workout", "meal", "habit", "metric"],
                            "description": "Type of event being logged"
                        },
                        "payload": {
                            "type": "object",
                            "description": "Event data payload"
                        }
                    },
                    "required": ["user_id", "event_type", "payload"]
                }
            ),
            "fetch_historical_data": MCPTool(
                name="fetch_historical_data",
                description="Fetch historical data for a specific metric or activity",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Unique identifier for the user"
                        },
                        "metric": {
                            "type": "string",
                            "description": "Metric type to fetch (weight, sleep, steps, etc.)"
                        },
                        "days": {
                            "type": "integer",
                            "default": 30,
                            "description": "Number of days to fetch data for"
                        }
                    },
                    "required": ["user_id", "metric"]
                }
            ),
            "send_notification": MCPTool(
                name="send_notification",
                description="Send notifications to users through various channels",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Unique identifier for the user"
                        },
                        "channel": {
                            "type": "string",
                            "enum": ["email", "push", "sms", "in_app"],
                            "description": "Notification channel"
                        },
                        "message": {
                            "type": "string",
                            "description": "Notification message content"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "default": "medium",
                            "description": "Notification priority level"
                        }
                    },
                    "required": ["user_id", "channel", "message"]
                }
            )
        }
    
    def _initialize_resources(self) -> Dict[str, MCPResource]:
        """Initialize MCP resources mapped from Supabase."""
        return {
            "user_goals": MCPResource(
                name="user_goals",
                description="User goals and objectives for health and fitness",
                uri="supabase://user_goals",
                mime_type="application/json"
            ),
            "health_metrics": MCPResource(
                name="health_metrics",
                description="User health metrics and measurements over time",
                uri="supabase://health_metrics",
                mime_type="application/json"
            ),
            "nutrition_database": MCPResource(
                name="nutrition_database",
                description="Comprehensive nutrition database with food information",
                uri="supabase://nutrition_database",
                mime_type="application/json"
            ),
            "exercise_library": MCPResource(
                name="exercise_library",
                description="Exercise library with detailed exercise information",
                uri="supabase://exercise_library",
                mime_type="application/json"
            )
        }
    
    def _initialize_prompts(self) -> Dict[str, MCPPrompt]:
        """Initialize reusable MCP prompts."""
        return {
            "plan_template": MCPPrompt(
                name="plan_template",
                description="Template for creating personalized fitness and health plans",
                template="""
Create a personalized {plan_type} plan for user with the following profile:
- Goals: {user_goals}
- Current metrics: {current_metrics}
- Preferences: {preferences}
- Constraints: {constraints}

The plan should be:
- Realistic and achievable
- Progressive and adaptive
- Aligned with user goals
- Safe and evidence-based

Include specific recommendations, timelines, and success metrics.
                """.strip(),
                variables=["plan_type", "user_goals", "current_metrics", "preferences", "constraints"]
            ),
            "diet_template": MCPPrompt(
                name="diet_template",
                description="Template for creating personalized nutrition plans",
                template="""
Create a personalized nutrition plan for user with:
- Caloric needs: {caloric_needs}
- Dietary restrictions: {restrictions}
- Goals: {nutrition_goals}
- Food preferences: {food_preferences}
- Activity level: {activity_level}

Provide:
- Daily meal structure
- Macro distribution
- Specific food recommendations
- Portion guidelines
- Meal timing suggestions
                """.strip(),
                variables=["caloric_needs", "restrictions", "nutrition_goals", "food_preferences", "activity_level"]
            ),
            "habit_template": MCPPrompt(
                name="habit_template",
                description="Template for creating habit formation recommendations",
                template="""
Suggest habits for user based on:
- Current lifestyle: {lifestyle}
- Available time: {time_availability}
- Goals: {habit_goals}
- Current habits: {existing_habits}
- Challenges: {challenges}

Provide:
- Specific habit recommendations
- Implementation strategies
- Habit stacking opportunities
- Progress tracking methods
- Motivation techniques
                """.strip(),
                variables=["lifestyle", "time_availability", "habit_goals", "existing_habits", "challenges"]
            ),
            "qol_template": MCPPrompt(
                name="qol_template",
                description="Template for quality of life assessment and recommendations",
                template="""
Assess quality of life for user with:
- Sleep patterns: {sleep_data}
- Stress levels: {stress_indicators}
- Social connections: {social_metrics}
- Work-life balance: {work_life_data}
- Physical health: {health_metrics}
- Mental wellbeing: {mental_health_data}

Provide:
- Overall QoL score
- Key improvement areas
- Specific recommendations
- Action priorities
- Progress tracking suggestions
                """.strip(),
                variables=["sleep_data", "stress_indicators", "social_metrics", "work_life_data", "health_metrics", "mental_health_data"]
            )
        }
    
    async def discover(self) -> Dict[str, Any]:
        """MCP discover endpoint - return available tools, resources, and prompts."""
        return {
            "tools": [tool.dict() for tool in self.tools.values()],
            "resources": [resource.dict() for resource in self.resources.values()],
            "prompts": [prompt.dict() for prompt in self.prompts.values()],
            "server_info": {
                "name": "Health & Quality of Life MCP Server",
                "version": "1.0.0",
                "description": "MCP server for orchestrating health and wellness AI agents"
            }
        }
    
    async def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get specific tool by name."""
        return self.tools.get(tool_name)
    
    async def get_resource(self, resource_name: str) -> Optional[Dict[str, Any]]:
        """Get specific resource data by name."""
        if resource_name not in self.resources:
            return None
        
        supabase = get_supabase()
        
        try:
            # Map resource name to Supabase table
            table_name = resource_name
            response = supabase.table(table_name).select("*").execute()
            
            return {
                "resource": self.resources[resource_name].dict(),
                "data": response.data
            }
        except Exception as e:
            logger.error(f"Error fetching resource {resource_name}: {e}")
            return None
    
    async def get_prompt(self, prompt_name: str) -> Optional[MCPPrompt]:
        """Get specific prompt by name."""
        return self.prompts.get(prompt_name)
    
    async def invoke_tool(self, request: MCPInvokeRequest) -> MCPInvokeResponse:
        """Invoke a specific tool with parameters."""
        tool_name = request.tool_name
        parameters = request.parameters
        
        try:
            if tool_name == "get_user_profile":
                result = await self._get_user_profile(parameters["user_id"])
            elif tool_name == "log_event":
                result = await self._log_event(
                    parameters["user_id"],
                    parameters["event_type"],
                    parameters["payload"]
                )
            elif tool_name == "fetch_historical_data":
                result = await self._fetch_historical_data(
                    parameters["user_id"],
                    parameters["metric"],
                    parameters.get("days", 30)
                )
            elif tool_name == "send_notification":
                result = await self._send_notification(
                    parameters["user_id"],
                    parameters["channel"],
                    parameters["message"],
                    parameters.get("priority", "medium")
                )
            else:
                return MCPInvokeResponse(
                    success=False,
                    result=None,
                    error=f"Unknown tool: {tool_name}"
                )
            
            return MCPInvokeResponse(success=True, result=result)
            
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            return MCPInvokeResponse(
                success=False,
                result=None,
                error=str(e)
            )
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile implementation."""
        supabase = get_supabase()
        
        # Fetch user goals
        goals_response = supabase.table("user_goals").select("*").eq("user_id", user_id).execute()
        
        # Fetch recent health metrics
        metrics_response = supabase.table("health_metrics").select("*").eq("user_id", user_id).order("recorded_at", desc=True).limit(10).execute()
        
        return {
            "user_id": user_id,
            "goals": goals_response.data,
            "recent_metrics": metrics_response.data,
            "profile_updated_at": datetime.utcnow().isoformat()
        }
    
    async def _log_event(self, user_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Log event implementation."""
        supabase = get_supabase()
        
        event_data = {
            "user_id": user_id,
            "event_type": event_type,
            "event_data": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = supabase.table("events_log").insert(event_data).execute()
        
        return {
            "event_logged": True,
            "event_id": response.data[0]["id"] if response.data else None,
            "timestamp": event_data["timestamp"]
        }
    
    async def _fetch_historical_data(self, user_id: str, metric: str, days: int) -> Dict[str, Any]:
        """Fetch historical data implementation."""
        supabase = get_supabase()
        
        # Calculate date range
        from datetime import timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        response = supabase.table("health_metrics").select("*").eq("user_id", user_id).eq("metric_type", metric).gte("recorded_at", start_date.isoformat()).lte("recorded_at", end_date.isoformat()).order("recorded_at", desc=False).execute()
        
        return {
            "user_id": user_id,
            "metric": metric,
            "days": days,
            "data_points": len(response.data),
            "data": response.data
        }
    
    async def _send_notification(self, user_id: str, channel: str, message: str, priority: str) -> Dict[str, Any]:
        """Send notification implementation (mock for now)."""
        # In a real implementation, this would integrate with notification services
        logger.info(f"Notification sent to {user_id} via {channel}: {message}")
        
        return {
            "notification_sent": True,
            "user_id": user_id,
            "channel": channel,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global MCP server instance
mcp_server = MCPServer()
