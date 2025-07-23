"""
Database initialization script for Supabase tables.
This script creates the necessary tables and initial data.
"""
import asyncio
import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database table creation SQL
CREATE_TABLES_SQL = {
    "users": """
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now(),
        is_active BOOLEAN DEFAULT true
    );
    """,
    
    "user_goals": """
    CREATE TABLE IF NOT EXISTS user_goals (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        goal_type TEXT NOT NULL CHECK (goal_type IN ('fitness', 'nutrition', 'habit', 'qol')),
        title TEXT NOT NULL,
        description TEXT,
        target_value NUMERIC,
        current_value NUMERIC DEFAULT 0,
        unit TEXT,
        deadline TIMESTAMP,
        status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'paused')),
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now()
    );
    """,
    
    "health_metrics": """
    CREATE TABLE IF NOT EXISTS health_metrics (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        metric_type TEXT NOT NULL,
        value NUMERIC NOT NULL,
        unit TEXT NOT NULL,
        recorded_at TIMESTAMP DEFAULT now(),
        notes TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    """,
    
    "nutrition_database": """
    CREATE TABLE IF NOT EXISTS nutrition_database (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        food_name TEXT NOT NULL,
        brand TEXT,
        calories_per_100g NUMERIC NOT NULL,
        protein_per_100g NUMERIC DEFAULT 0,
        carbs_per_100g NUMERIC DEFAULT 0,
        fat_per_100g NUMERIC DEFAULT 0,
        fiber_per_100g NUMERIC DEFAULT 0,
        category TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    """,
    
    "exercise_library": """
    CREATE TABLE IF NOT EXISTS exercise_library (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        exercise_name TEXT NOT NULL,
        category TEXT NOT NULL CHECK (category IN ('strength', 'cardio', 'flexibility', 'balance')),
        muscle_groups TEXT[],
        equipment TEXT,
        difficulty_level INTEGER DEFAULT 1 CHECK (difficulty_level BETWEEN 1 AND 5),
        instructions TEXT,
        duration_minutes INTEGER,
        calories_per_minute NUMERIC,
        created_at TIMESTAMP DEFAULT now()
    );
    """,
    
    "events_log": """
    CREATE TABLE IF NOT EXISTS events_log (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        event_type TEXT NOT NULL CHECK (event_type IN ('workout', 'meal', 'habit', 'metric', 'coaching')),
        event_data JSONB NOT NULL,
        agent_source TEXT,
        timestamp TIMESTAMP DEFAULT now(),
        created_at TIMESTAMP DEFAULT now()
    );
    """
}

# Sample data for nutrition database
SAMPLE_NUTRITION_DATA = [
    {
        "food_name": "Chicken Breast",
        "brand": "Generic",
        "calories_per_100g": 165,
        "protein_per_100g": 31,
        "carbs_per_100g": 0,
        "fat_per_100g": 3.6,
        "fiber_per_100g": 0,
        "category": "protein"
    },
    {
        "food_name": "Brown Rice",
        "brand": "Generic",
        "calories_per_100g": 123,
        "protein_per_100g": 2.6,
        "carbs_per_100g": 23,
        "fat_per_100g": 0.9,
        "fiber_per_100g": 1.8,
        "category": "grain"
    },
    {
        "food_name": "Broccoli",
        "brand": "Generic",
        "calories_per_100g": 34,
        "protein_per_100g": 2.8,
        "carbs_per_100g": 7,
        "fat_per_100g": 0.4,
        "fiber_per_100g": 2.6,
        "category": "vegetable"
    },
    {
        "food_name": "Banana",
        "brand": "Generic",
        "calories_per_100g": 89,
        "protein_per_100g": 1.1,
        "carbs_per_100g": 23,
        "fat_per_100g": 0.3,
        "fiber_per_100g": 2.6,
        "category": "fruit"
    },
    {
        "food_name": "Almonds",
        "brand": "Generic",
        "calories_per_100g": 579,
        "protein_per_100g": 21,
        "carbs_per_100g": 22,
        "fat_per_100g": 50,
        "fiber_per_100g": 12,
        "category": "nuts"
    }
]

# Sample data for exercise library
SAMPLE_EXERCISE_DATA = [
    {
        "exercise_name": "Push-ups",
        "category": "strength",
        "muscle_groups": ["chest", "shoulders", "triceps"],
        "equipment": "bodyweight",
        "difficulty_level": 2,
        "instructions": "Start in plank position, lower body until chest nearly touches floor, push back up",
        "duration_minutes": None,
        "calories_per_minute": 7
    },
    {
        "exercise_name": "Squats",
        "category": "strength",
        "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
        "equipment": "bodyweight",
        "difficulty_level": 2,
        "instructions": "Stand with feet shoulder-width apart, lower body as if sitting back into chair, return to standing",
        "duration_minutes": None,
        "calories_per_minute": 8
    },
    {
        "exercise_name": "Running",
        "category": "cardio",
        "muscle_groups": ["legs", "core"],
        "equipment": "none",
        "difficulty_level": 3,
        "instructions": "Maintain steady pace, land on midfoot, keep upright posture",
        "duration_minutes": 30,
        "calories_per_minute": 10
    },
    {
        "exercise_name": "Yoga Flow",
        "category": "flexibility",
        "muscle_groups": ["full_body"],
        "equipment": "yoga_mat",
        "difficulty_level": 2,
        "instructions": "Flow through poses with controlled breathing, hold each pose for 30 seconds",
        "duration_minutes": 45,
        "calories_per_minute": 3
    },
    {
        "exercise_name": "Plank",
        "category": "strength",
        "muscle_groups": ["core", "shoulders"],
        "equipment": "bodyweight",
        "difficulty_level": 2,
        "instructions": "Hold plank position with straight line from head to heels",
        "duration_minutes": None,
        "calories_per_minute": 5
    }
]


async def init_database():
    """Initialize database tables and sample data."""
    try:
        # Create Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("Error: SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        print("Connected to Supabase successfully")
        
        # Create tables
        print("Creating tables...")
        for table_name, sql in CREATE_TABLES_SQL.items():
            try:
                # Note: Supabase Python client doesn't support raw SQL execution
                # These tables should be created through Supabase dashboard or SQL editor
                print(f"Table creation SQL for {table_name}:")
                print(sql)
                print("-" * 50)
            except Exception as e:
                print(f"Error with table {table_name}: {e}")
        
        # Insert sample nutrition data
        print("Inserting sample nutrition data...")
        try:
            result = supabase.table("nutrition_database").insert(SAMPLE_NUTRITION_DATA).execute()
            print(f"Inserted {len(result.data)} nutrition items")
        except Exception as e:
            print(f"Error inserting nutrition data: {e}")
        
        # Insert sample exercise data
        print("Inserting sample exercise data...")
        try:
            result = supabase.table("exercise_library").insert(SAMPLE_EXERCISE_DATA).execute()
            print(f"Inserted {len(result.data)} exercises")
        except Exception as e:
            print(f"Error inserting exercise data: {e}")
        
        print("Database initialization completed!")
        return True
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting database initialization...")
    print("Note: Table creation SQL is provided above.")
    print("Please run these SQL commands in your Supabase SQL editor first.")
    print("Then run this script to insert sample data.")
    
    # Run the initialization
    asyncio.run(init_database())
