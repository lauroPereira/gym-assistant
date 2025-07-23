"""
Training Agent - Generates personalized fitness and training plans.
"""
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from agents.base_agent import BaseMCPAgent
from models.schemas import TrainingPlan
import logging

logger = logging.getLogger(__name__)


class TrainingAgent(BaseMCPAgent):
    """Agent responsible for creating and managing training plans."""
    
    def __init__(self):
        super().__init__("TrainingAgent")
    
    async def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a personalized training plan for the user."""
        try:
            # Get user profile and goals
            user_profile = await self.get_user_profile(user_id)
            
            # Get exercise library
            exercise_data = await self.get_resource_data("exercise_library")
            exercises = exercise_data.get("data", []) if exercise_data else []
            
            # Get historical workout data
            workout_history = await self.fetch_historical_data(user_id, "workout", 30)
            
            # Get fitness goals
            fitness_goals = [
                goal for goal in user_profile.get("goals", [])
                if goal.get("goal_type") == "fitness"
            ]
            
            # Generate training plan using prompt template
            plan_template = await self.get_prompt_template("plan_template")
            if plan_template:
                formatted_prompt = self.format_prompt(plan_template, {
                    "plan_type": "fitness training",
                    "user_goals": json.dumps(fitness_goals),
                    "current_metrics": json.dumps(user_profile.get("recent_metrics", [])),
                    "preferences": json.dumps(context.get("preferences", {})),
                    "constraints": json.dumps(context.get("constraints", {}))
                })
            
            # Create training plan based on user data
            training_plan = await self._create_training_plan(
                user_id, fitness_goals, exercises, workout_history, context
            )
            
            # Log the training plan creation
            await self.log_event(user_id, "workout", {
                "action": "training_plan_created",
                "plan_id": training_plan["plan_id"],
                "duration_weeks": training_plan["duration_weeks"]
            })
            
            return {
                "agent": self.agent_name,
                "success": True,
                "training_plan": training_plan,
                "recommendations": await self._get_training_recommendations(user_id, training_plan),
                "next_actions": [
                    "Start with the beginner exercises",
                    "Track your progress daily",
                    "Adjust intensity based on your comfort level"
                ]
            }
            
        except Exception as e:
            logger.error(f"TrainingAgent process error: {e}")
            return {
                "agent": self.agent_name,
                "success": False,
                "error": str(e),
                "training_plan": None
            }
    
    async def _create_training_plan(
        self, 
        user_id: str, 
        goals: List[Dict], 
        exercises: List[Dict], 
        history: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Create a personalized training plan."""
        
        # Determine user fitness level
        fitness_level = self._assess_fitness_level(history, context)
        
        # Select appropriate exercises
        selected_exercises = self._select_exercises(exercises, goals, fitness_level)
        
        # Create weekly schedule
        weekly_schedule = self._create_weekly_schedule(selected_exercises, fitness_level)
        
        plan_id = f"training_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        return {
            "plan_id": plan_id,
            "user_id": user_id,
            "exercises": selected_exercises,
            "weekly_schedule": weekly_schedule,
            "duration_weeks": 8,  # Default 8-week program
            "frequency_per_week": len(weekly_schedule),
            "difficulty_level": fitness_level,
            "goals": [goal.get("title", "") for goal in goals],
            "notes": "Progressive training plan tailored to your goals and fitness level",
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _assess_fitness_level(self, history: Dict, context: Dict) -> int:
        """Assess user fitness level (1-5 scale)."""
        # Simple assessment based on workout history and self-reported level
        workout_count = len(history.get("data", []))
        
        if workout_count == 0:
            return 1  # Beginner
        elif workout_count < 10:
            return 2  # Novice
        elif workout_count < 30:
            return 3  # Intermediate
        elif workout_count < 60:
            return 4  # Advanced
        else:
            return 5  # Expert
    
    def _select_exercises(self, exercises: List[Dict], goals: List[Dict], fitness_level: int) -> List[Dict[str, Any]]:
        """Select appropriate exercises based on goals and fitness level."""
        selected = []
        
        # Filter exercises by difficulty level
        suitable_exercises = [
            ex for ex in exercises 
            if ex.get("difficulty_level", 1) <= fitness_level + 1
        ]
        
        # Categorize exercises
        strength_exercises = [ex for ex in suitable_exercises if ex.get("category") == "strength"]
        cardio_exercises = [ex for ex in suitable_exercises if ex.get("category") == "cardio"]
        flexibility_exercises = [ex for ex in suitable_exercises if ex.get("category") == "flexibility"]
        
        # Select based on goals
        goal_keywords = " ".join([goal.get("title", "").lower() for goal in goals])
        
        if "strength" in goal_keywords or "muscle" in goal_keywords:
            selected.extend(strength_exercises[:6])
        if "cardio" in goal_keywords or "endurance" in goal_keywords:
            selected.extend(cardio_exercises[:4])
        if "flexibility" in goal_keywords or "mobility" in goal_keywords:
            selected.extend(flexibility_exercises[:3])
        
        # Default selection if no specific goals
        if not selected:
            selected.extend(strength_exercises[:3])
            selected.extend(cardio_exercises[:2])
            selected.extend(flexibility_exercises[:2])
        
        # Add sets, reps, and progression
        for exercise in selected:
            exercise.update(self._get_exercise_prescription(exercise, fitness_level))
        
        return selected[:10]  # Limit to 10 exercises
    
    def _get_exercise_prescription(self, exercise: Dict, fitness_level: int) -> Dict[str, Any]:
        """Get sets, reps, and other prescription details for an exercise."""
        category = exercise.get("category", "strength")
        
        if category == "strength":
            return {
                "sets": min(2 + fitness_level, 5),
                "reps": f"{8 + fitness_level * 2}-{12 + fitness_level * 2}",
                "rest_seconds": 60 + fitness_level * 15,
                "progression": "Increase weight or reps weekly"
            }
        elif category == "cardio":
            return {
                "duration_minutes": 15 + fitness_level * 5,
                "intensity": f"Level {fitness_level}/5",
                "progression": "Increase duration by 2-3 minutes weekly"
            }
        else:  # flexibility
            return {
                "duration_minutes": 10 + fitness_level * 2,
                "hold_seconds": 20 + fitness_level * 5,
                "progression": "Increase hold time and range of motion"
            }
    
    def _create_weekly_schedule(self, exercises: List[Dict], fitness_level: int) -> List[Dict[str, Any]]:
        """Create a weekly workout schedule."""
        days_per_week = min(3 + fitness_level, 6)
        
        schedule = []
        exercise_groups = {
            "strength": [ex for ex in exercises if ex.get("category") == "strength"],
            "cardio": [ex for ex in exercises if ex.get("category") == "cardio"],
            "flexibility": [ex for ex in exercises if ex.get("category") == "flexibility"]
        }
        
        for day in range(days_per_week):
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day]
            
            if day % 3 == 0:  # Strength days
                day_exercises = exercise_groups["strength"][:4]
                workout_type = "Strength Training"
            elif day % 3 == 1:  # Cardio days
                day_exercises = exercise_groups["cardio"][:2] + exercise_groups["flexibility"][:1]
                workout_type = "Cardio & Flexibility"
            else:  # Mixed days
                day_exercises = exercise_groups["strength"][:2] + exercise_groups["cardio"][:1]
                workout_type = "Mixed Training"
            
            schedule.append({
                "day": day_name,
                "workout_type": workout_type,
                "exercises": day_exercises,
                "estimated_duration": 45 + len(day_exercises) * 5
            })
        
        return schedule
    
    async def _get_training_recommendations(self, user_id: str, training_plan: Dict) -> List[str]:
        """Get personalized training recommendations."""
        recommendations = [
            "Warm up for 5-10 minutes before each workout",
            "Focus on proper form over heavy weights",
            "Stay hydrated throughout your workout",
            "Get adequate rest between training sessions"
        ]
        
        difficulty = training_plan.get("difficulty_level", 1)
        
        if difficulty <= 2:
            recommendations.extend([
                "Start with bodyweight exercises to build foundation",
                "Progress gradually to avoid injury",
                "Consider working with a trainer initially"
            ])
        else:
            recommendations.extend([
                "Challenge yourself with progressive overload",
                "Track your performance metrics",
                "Incorporate periodization in your training"
            ])
        
        return recommendations
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get TrainingAgent information."""
        return {
            "name": self.agent_name,
            "description": "Generates personalized fitness and training plans",
            "capabilities": [
                "Fitness level assessment",
                "Exercise selection and prescription",
                "Weekly schedule creation",
                "Progress tracking recommendations",
                "Training plan adaptation"
            ],
            "endpoints": [
                "/api/agents/training/plan",
                "/api/agents/training/progress",
                "/api/agents/training/adjust"
            ]
        }
