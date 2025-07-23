"""
Base MCP Agent class for the Health & Quality of Life App.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
from backend.mcp.server import mcp_server
from backend.models.schemas import MCPInvokeRequest

logger = logging.getLogger(__name__)


class BaseMCPAgent(ABC):
    """Base class for MCP agents that consume tools, resources, and prompts."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.mcp_server = mcp_server
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile using MCP tool."""
        request = MCPInvokeRequest(
            tool_name="get_user_profile",
            parameters={"user_id": user_id}
        )
        response = await self.mcp_server.invoke_tool(request)
        return response.result if response.success else {}
    
    async def log_event(self, user_id: str, event_type: str, payload: Dict[str, Any]) -> bool:
        """Log event using MCP tool."""
        request = MCPInvokeRequest(
            tool_name="log_event",
            parameters={
                "user_id": user_id,
                "event_type": event_type,
                "payload": {**payload, "agent_source": self.agent_name}
            }
        )
        response = await self.mcp_server.invoke_tool(request)
        return response.success
    
    async def fetch_historical_data(self, user_id: str, metric: str, days: int = 30) -> Dict[str, Any]:
        """Fetch historical data using MCP tool."""
        request = MCPInvokeRequest(
            tool_name="fetch_historical_data",
            parameters={
                "user_id": user_id,
                "metric": metric,
                "days": days
            }
        )
        response = await self.mcp_server.invoke_tool(request)
        return response.result if response.success else {}
    
    async def send_notification(self, user_id: str, channel: str, message: str, priority: str = "medium") -> bool:
        """Send notification using MCP tool."""
        request = MCPInvokeRequest(
            tool_name="send_notification",
            parameters={
                "user_id": user_id,
                "channel": channel,
                "message": message,
                "priority": priority
            }
        )
        response = await self.mcp_server.invoke_tool(request)
        return response.success
    
    async def get_resource_data(self, resource_name: str) -> Dict[str, Any]:
        """Get resource data using MCP server."""
        return await self.mcp_server.get_resource(resource_name)
    
    async def get_prompt_template(self, prompt_name: str) -> Optional[str]:
        """Get prompt template using MCP server."""
        prompt = await self.mcp_server.get_prompt(prompt_name)
        return prompt.template if prompt else None
    
    def format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """Format prompt template with variables."""
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in prompt template: {e}")
            return template
    
    @abstractmethod
    async def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent-specific logic. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information. Must be implemented by subclasses."""
        pass
