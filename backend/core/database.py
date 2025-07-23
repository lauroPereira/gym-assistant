"""
Database configuration and initialization for Supabase.
"""
from supabase import create_client, Client
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Global Supabase client
supabase: Client = None


async def init_db():
    """Initialize Supabase client and database connection."""
    global supabase
    
    try:
        supabase = create_client(settings.supabase_url, settings.supabase_key)
        logger.info("Supabase client initialized successfully")
        
        # Test connection
        response = supabase.table("user_goals").select("*").limit(1).execute()
        logger.info("Database connection test successful")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_supabase() -> Client:
    """Get Supabase client instance."""
    if supabase is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return supabase


# Database schemas for reference
DATABASE_SCHEMAS = {
    "user_goals": {
        "id": "uuid PRIMARY KEY DEFAULT gen_random_uuid()",
        "user_id": "uuid NOT NULL",
        "goal_type": "text NOT NULL",  # 'fitness', 'nutrition', 'habit', 'qol'
        "title": "text NOT NULL",
        "description": "text",
        "target_value": "numeric",
        "current_value": "numeric DEFAULT 0",
        "unit": "text",
        "deadline": "timestamp",
        "status": "text DEFAULT 'active'",  # 'active', 'completed', 'paused'
        "created_at": "timestamp DEFAULT now()",
        "updated_at": "timestamp DEFAULT now()"
    },
    
    "health_metrics": {
        "id": "uuid PRIMARY KEY DEFAULT gen_random_uuid()",
        "user_id": "uuid NOT NULL",
        "metric_type": "text NOT NULL",  # 'weight', 'sleep', 'steps', 'heart_rate', etc.
        "value": "numeric NOT NULL",
        "unit": "text NOT NULL",
        "recorded_at": "timestamp DEFAULT now()",
        "notes": "text",
        "created_at": "timestamp DEFAULT now()"
    },
    
    "nutrition_database": {
        "id": "uuid PRIMARY KEY DEFAULT gen_random_uuid()",
        "food_name": "text NOT NULL",
        "brand": "text",
        "calories_per_100g": "numeric NOT NULL",
        "protein_per_100g": "numeric DEFAULT 0",
        "carbs_per_100g": "numeric DEFAULT 0",
        "fat_per_100g": "numeric DEFAULT 0",
        "fiber_per_100g": "numeric DEFAULT 0",
        "category": "text",  # 'protein', 'vegetable', 'fruit', 'grain', etc.
        "created_at": "timestamp DEFAULT now()"
    },
    
    "exercise_library": {
        "id": "uuid PRIMARY KEY DEFAULT gen_random_uuid()",
        "exercise_name": "text NOT NULL",
        "category": "text NOT NULL",  # 'strength', 'cardio', 'flexibility', 'balance'
        "muscle_groups": "text[]",  # Array of muscle groups
        "equipment": "text",
        "difficulty_level": "integer DEFAULT 1",  # 1-5 scale
        "instructions": "text",
        "duration_minutes": "integer",
        "calories_per_minute": "numeric",
        "created_at": "timestamp DEFAULT now()"
    },
    
    "events_log": {
        "id": "uuid PRIMARY KEY DEFAULT gen_random_uuid()",
        "user_id": "uuid NOT NULL",
        "event_type": "text NOT NULL",  # 'workout', 'meal', 'habit', 'metric'
        "event_data": "jsonb NOT NULL",
        "agent_source": "text",  # Which agent logged this event
        "timestamp": "timestamp DEFAULT now()",
        "created_at": "timestamp DEFAULT now()"
    }
}
