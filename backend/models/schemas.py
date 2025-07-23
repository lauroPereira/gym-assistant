"""
Pydantic models and schemas for the Health & Quality of Life MCP App.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class GoalType(str, Enum):
    FITNESS = "fitness"
    NUTRITION = "nutrition"
    HABIT = "habit"
    QOL = "qol"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"


class MetricType(str, Enum):
    WEIGHT = "weight"
    SLEEP = "sleep"
    STEPS = "steps"
    HEART_RATE = "heart_rate"
    MOOD = "mood"
    ENERGY = "energy"


class EventType(str, Enum):
    WORKOUT = "workout"
    MEAL = "meal"
    HABIT = "habit"
    METRIC = "metric"


# Authentication schemas
class UserLogin(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


class UserRegister(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password")
    full_name: Optional[str] = Field(None, description="User full name")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class User(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None


# MCP schemas
class MCPTool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class MCPResource(BaseModel):
    name: str
    description: str
    uri: str
    mime_type: str


class MCPPrompt(BaseModel):
    name: str
    description: str
    template: str
    variables: List[str]


class MCPInvokeRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class MCPInvokeResponse(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None


# Health and fitness schemas
class UserGoal(BaseModel):
    id: Optional[str] = None
    user_id: str
    goal_type: GoalType
    title: str
    description: Optional[str] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = 0
    unit: Optional[str] = None
    deadline: Optional[datetime] = None
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class HealthMetric(BaseModel):
    id: Optional[str] = None
    user_id: str
    metric_type: MetricType
    value: float
    unit: str
    recorded_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None


class NutritionItem(BaseModel):
    id: Optional[str] = None
    food_name: str
    brand: Optional[str] = None
    calories_per_100g: float
    protein_per_100g: Optional[float] = 0
    carbs_per_100g: Optional[float] = 0
    fat_per_100g: Optional[float] = 0
    fiber_per_100g: Optional[float] = 0
    category: Optional[str] = None
    created_at: Optional[datetime] = None


class Exercise(BaseModel):
    id: Optional[str] = None
    exercise_name: str
    category: str
    muscle_groups: List[str]
    equipment: Optional[str] = None
    difficulty_level: int = Field(default=1, ge=1, le=5)
    instructions: Optional[str] = None
    duration_minutes: Optional[int] = None
    calories_per_minute: Optional[float] = None
    created_at: Optional[datetime] = None


class EventLog(BaseModel):
    id: Optional[str] = None
    user_id: str
    event_type: EventType
    event_data: Dict[str, Any]
    agent_source: Optional[str] = None
    timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None


# Agent response schemas
class TrainingPlan(BaseModel):
    plan_id: str
    user_id: str
    exercises: List[Dict[str, Any]]
    duration_weeks: int
    frequency_per_week: int
    difficulty_level: int
    goals: List[str]
    notes: Optional[str] = None


class DietPlan(BaseModel):
    plan_id: str
    user_id: str
    daily_calories: int
    meals: List[Dict[str, Any]]
    macros: Dict[str, float]
    restrictions: List[str]
    duration_days: int
    notes: Optional[str] = None


class HabitSuggestion(BaseModel):
    habit_id: str
    user_id: str
    habit_name: str
    description: str
    frequency: str
    difficulty: int
    category: str
    benefits: List[str]
    implementation_tips: List[str]


class QoLMetrics(BaseModel):
    user_id: str
    overall_score: float
    sleep_quality: float
    stress_level: float
    energy_level: float
    mood_score: float
    social_connections: float
    work_life_balance: float
    recommendations: List[str]


class OrchestratorResponse(BaseModel):
    user_id: str
    timestamp: datetime
    training_plan: Optional[TrainingPlan] = None
    diet_plan: Optional[DietPlan] = None
    habit_suggestions: List[HabitSuggestion] = []
    qol_metrics: Optional[QoLMetrics] = None
    summary: str
    next_actions: List[str]
