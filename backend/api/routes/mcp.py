"""
MCP Server API routes for the Health & Quality of Life MCP App.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from backend.models.schemas import MCPInvokeRequest, MCPInvokeResponse
from backend.mcp.server import mcp_server
from backend.core.auth import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/discover")
async def discover():
    """MCP discover endpoint - return available tools, resources, and prompts."""
    try:
        return await mcp_server.discover()
    except Exception as e:
        logger.error(f"MCP discover error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to discover MCP capabilities"
        )


@router.get("/tools/{tool_name}")
async def get_tool(tool_name: str):
    """Get specific MCP tool by name."""
    try:
        tool = await mcp_server.get_tool(tool_name)
        if tool is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found"
            )
        return tool.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get tool error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tool"
        )


@router.get("/resources/{resource_name}")
async def get_resource(resource_name: str, current_user: dict = Depends(get_current_user)):
    """Get specific MCP resource by name."""
    try:
        resource = await mcp_server.get_resource(resource_name)
        if resource is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resource '{resource_name}' not found"
            )
        return resource
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get resource error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve resource"
        )


@router.get("/prompts/{prompt_name}")
async def get_prompt(prompt_name: str):
    """Get specific MCP prompt by name."""
    try:
        prompt = await mcp_server.get_prompt(prompt_name)
        if prompt is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt '{prompt_name}' not found"
            )
        return prompt.dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get prompt error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prompt"
        )


@router.post("/invoke", response_model=MCPInvokeResponse)
async def invoke_tool(request: MCPInvokeRequest, current_user: dict = Depends(get_current_user)):
    """Invoke an MCP tool with parameters."""
    try:
        # Add user context to parameters if not present
        if "user_id" not in request.parameters:
            request.parameters["user_id"] = current_user["user_id"]
        
        response = await mcp_server.invoke_tool(request)
        return response
    except Exception as e:
        logger.error(f"Tool invocation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to invoke tool"
        )


@router.get("/status")
async def mcp_status():
    """Get MCP server status and capabilities."""
    try:
        capabilities = await mcp_server.discover()
        return {
            "status": "healthy",
            "tools_count": len(capabilities["tools"]),
            "resources_count": len(capabilities["resources"]),
            "prompts_count": len(capabilities["prompts"]),
            "server_info": capabilities["server_info"]
        }
    except Exception as e:
        logger.error(f"MCP status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP status"
        )
