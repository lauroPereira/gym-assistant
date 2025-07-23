"""
Main FastAPI application entry point for the Health & Quality of Life MCP App.
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import os
from dotenv import load_dotenv

from core.config import settings
from api.routes import mcp, auth, orchestrator
from core.database import init_db

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Health & Quality of Life MCP App",
    description="A health application using Model Context Protocol to orchestrate multiple AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(mcp.router, prefix="/api/mcp", tags=["MCP Server"])
app.include_router(orchestrator.router, prefix="/api/orchestrator", tags=["Orchestrator"])

@app.on_event("startup")
async def startup_event():
    """Initialize database and other startup tasks."""
    await init_db()

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Health & Quality of Life MCP App",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
