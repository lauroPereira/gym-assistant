"""
Habit Agent - Generates personalized habit formation recommendations.
"""
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from agents.base_agent import BaseMCPAgent
from models.schemas import HabitSuggestion
import logging

logger = logging.getLogger(__name__)


class HabitAgent(BaseMCPAgent):
    """Agent responsible for creating and managing habit recommendations."""
    
    def __init__(self):
        super().__init__("HabitAgent")
    
    async def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized habit suggestions for the user."""
        try:
            # Get user profile and goals
            user_profile = await self.get_user_profile(user_id)
            
            # Get historical habit data
            habit_history = await self.fetch_historical_data(user_id, "habit", 30)
            
            # Get habit-related goals
            habit_goals = [
                goal for goal in user_profile.get("goals", [])
                if goal.get("goal_type") == "habit"
            ]
            
            # Generate habit suggestions using prompt template
            habit_template = await self.get_prompt_template("habit_template")
            if habit_template:
                formatted_prompt = self.format_prompt(habit_template, {
                    "lifestyle": json.dumps(context.get("lifestyle", {})),
                    "time_availability": context.get("time_availability", "moderate"),
                    "habit_goals": json.dumps(habit_goals),
                    "existing_habits": json.dumps(habit_history.get("data", [])),
                    "challenges": json.dumps(context.get("challenges", []))
                })
            
            # Create habit suggestions
            habit_suggestions = await self._create_habit_suggestions(
                user_id, habit_goals, habit_history, context
            )
            
            # Log the habit suggestions creation
            await self.log_event(user_id, "habit", {
                "action": "habit_suggestions_created",
                "suggestions_count": len(habit_suggestions),
                "categories": list(set(h["category"] for h in habit_suggestions))
            })
            
            return {
                "agent": self.agent_name,
                "success": True,
                "habit_suggestions": habit_suggestions,
                "recommendations": await self._get_habit_recommendations(user_id, habit_suggestions),
                "next_actions": [
                    "Choose 1-2 habits to start with",
                    "Set up habit tracking system",
                    "Create environmental cues for new habits"
                ]
            }
            
        except Exception as e:
            logger.error(f"HabitAgent process error: {e}")
            return {
                "agent": self.agent_name,
                "success": False,
                "error": str(e),
                "habit_suggestions": []
            }
    
    async def _create_habit_suggestions(
        self, 
        user_id: str, 
        goals: List[Dict], 
        history: Dict, 
        context: Dict
    ) -> List[Dict[str, Any]]:
        """Create personalized habit suggestions."""
        
        suggestions = []
        
        # Define habit categories and their suggestions
        habit_categories = {
            "health": self._get_health_habits(),
            "productivity": self._get_productivity_habits(),
            "wellness": self._get_wellness_habits(),
            "fitness": self._get_fitness_habits(),
            "nutrition": self._get_nutrition_habits()
        }
        
        # Analyze user goals to determine relevant categories
        goal_text = " ".join([goal.get("title", "").lower() for goal in goals])
        existing_habits = [h.get("event_data", {}).get("habit_name", "") for h in history.get("data", [])]
        
        # Select habits based on goals and avoid duplicates
        for category, category_habits in habit_categories.items():
            if self._is_category_relevant(category, goal_text, context):
                for habit in category_habits:
                    if habit["habit_name"] not in existing_habits:
                        habit_suggestion = self._create_habit_suggestion(user_id, habit, context)
                        suggestions.append(habit_suggestion)
        
        # Sort by difficulty and relevance
        suggestions.sort(key=lambda x: (x["difficulty"], -len(x["benefits"])))
        
        return suggestions[:8]  # Limit to 8 suggestions
    
    def _get_health_habits(self) -> List[Dict[str, Any]]:
        """Get health-related habit suggestions."""
        return [
            {
                "habit_name": "Drink 8 glasses of water daily",
                "description": "Stay hydrated by drinking water throughout the day",
                "frequency": "daily",
                "difficulty": 2,
                "category": "health",
                "benefits": ["Better hydration", "Improved energy", "Better skin health"],
                "implementation_tips": ["Keep water bottle visible", "Set hourly reminders", "Track intake"]
            },
            {
                "habit_name": "Take 10,000 steps daily",
                "description": "Increase daily movement through walking",
                "frequency": "daily",
                "difficulty": 3,
                "category": "health",
                "benefits": ["Improved cardiovascular health", "Better mood", "Weight management"],
                "implementation_tips": ["Use step counter", "Take stairs", "Walk during breaks"]
            },
            {
                "habit_name": "Sleep 7-8 hours nightly",
                "description": "Maintain consistent sleep schedule",
                "frequency": "daily",
                "difficulty": 3,
                "category": "health",
                "benefits": ["Better recovery", "Improved focus", "Better immune system"],
                "implementation_tips": ["Set bedtime alarm", "Avoid screens before bed", "Create bedtime routine"]
            }
        ]
    
    def _get_productivity_habits(self) -> List[Dict[str, Any]]:
        """Get productivity-related habit suggestions."""
        return [
            {
                "habit_name": "Plan tomorrow tonight",
                "description": "Spend 10 minutes planning the next day",
                "frequency": "daily",
                "difficulty": 2,
                "category": "productivity",
                "benefits": ["Better time management", "Reduced stress", "Clear priorities"],
                "implementation_tips": ["Use planning app", "Review goals", "Set top 3 priorities"]
            },
            {
                "habit_name": "Single-task focus",
                "description": "Focus on one task at a time without distractions",
                "frequency": "daily",
                "difficulty": 4,
                "category": "productivity",
                "benefits": ["Better quality work", "Faster completion", "Less stress"],
                "implementation_tips": ["Turn off notifications", "Use timer", "Clear workspace"]
            }
        ]
    
    def _get_wellness_habits(self) -> List[Dict[str, Any]]:
        """Get wellness-related habit suggestions."""
        return [
            {
                "habit_name": "5-minute meditation",
                "description": "Practice mindfulness meditation daily",
                "frequency": "daily",
                "difficulty": 2,
                "category": "wellness",
                "benefits": ["Reduced stress", "Better focus", "Emotional regulation"],
                "implementation_tips": ["Use meditation app", "Same time daily", "Quiet space"]
            },
            {
                "habit_name": "Gratitude journaling",
                "description": "Write 3 things you're grateful for each day",
                "frequency": "daily",
                "difficulty": 1,
                "category": "wellness",
                "benefits": ["Positive mindset", "Better mood", "Improved relationships"],
                "implementation_tips": ["Keep journal by bed", "Be specific", "Include small things"]
            },
            {
                "habit_name": "Digital sunset",
                "description": "No screens 1 hour before bedtime",
                "frequency": "daily",
                "difficulty": 4,
                "category": "wellness",
                "benefits": ["Better sleep quality", "Reduced eye strain", "More relaxation"],
                "implementation_tips": ["Use blue light filters", "Read instead", "Charge phone outside bedroom"]
            }
        ]
    
    def _get_fitness_habits(self) -> List[Dict[str, Any]]:
        """Get fitness-related habit suggestions."""
        return [
            {
                "habit_name": "Morning stretching",
                "description": "10-minute stretching routine each morning",
                "frequency": "daily",
                "difficulty": 2,
                "category": "fitness",
                "benefits": ["Better flexibility", "Reduced stiffness", "Injury prevention"],
                "implementation_tips": ["Follow video routine", "Focus on tight areas", "Breathe deeply"]
            },
            {
                "habit_name": "Workout consistency",
                "description": "Exercise at the same time each day",
                "frequency": "daily",
                "difficulty": 3,
                "category": "fitness",
                "benefits": ["Better routine", "Improved results", "Habit formation"],
                "implementation_tips": ["Schedule workouts", "Prepare gear ahead", "Start small"]
            }
        ]
    
    def _get_nutrition_habits(self) -> List[Dict[str, Any]]:
        """Get nutrition-related habit suggestions."""
        return [
            {
                "habit_name": "Eat vegetables first",
                "description": "Start each meal with vegetables",
                "frequency": "daily",
                "difficulty": 2,
                "category": "nutrition",
                "benefits": ["Better nutrition", "Increased satiety", "Weight management"],
                "implementation_tips": ["Prep vegetables ahead", "Keep them visible", "Try different varieties"]
            },
            {
                "habit_name": "Mindful eating",
                "description": "Eat without distractions, focus on food",
                "frequency": "daily",
                "difficulty": 3,
                "category": "nutrition",
                "benefits": ["Better digestion", "Portion control", "Food appreciation"],
                "implementation_tips": ["No phones during meals", "Chew slowly", "Notice flavors"]
            }
        ]
    
    def _is_category_relevant(self, category: str, goal_text: str, context: Dict) -> bool:
        """Determine if a habit category is relevant to user goals."""
        category_keywords = {
            "health": ["health", "wellness", "energy", "vitality"],
            "productivity": ["productivity", "work", "focus", "efficiency"],
            "wellness": ["stress", "mental", "mindfulness", "balance"],
            "fitness": ["fitness", "exercise", "strength", "cardio"],
            "nutrition": ["nutrition", "diet", "eating", "weight"]
        }
        
        keywords = category_keywords.get(category, [])
        return any(keyword in goal_text for keyword in keywords)
    
    def _create_habit_suggestion(self, user_id: str, habit: Dict, context: Dict) -> Dict[str, Any]:
        """Create a habit suggestion with personalized details."""
        habit_id = f"habit_{user_id}_{habit['habit_name'].replace(' ', '_')}_{int(datetime.utcnow().timestamp())}"
        
        return {
            "habit_id": habit_id,
            "user_id": user_id,
            "habit_name": habit["habit_name"],
            "description": habit["description"],
            "frequency": habit["frequency"],
            "difficulty": habit["difficulty"],
            "category": habit["category"],
            "benefits": habit["benefits"],
            "implementation_tips": habit["implementation_tips"],
            "estimated_time_minutes": self._estimate_habit_time(habit),
            "success_rate": self._estimate_success_rate(habit, context),
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _estimate_habit_time(self, habit: Dict) -> int:
        """Estimate time required for habit in minutes."""
        time_estimates = {
            "meditation": 5,
            "stretching": 10,
            "journaling": 5,
            "planning": 10,
            "water": 1,
            "steps": 30,
            "sleep": 0,  # No additional time
            "eating": 0   # Part of existing meals
        }
        
        habit_name = habit["habit_name"].lower()
        for keyword, time in time_estimates.items():
            if keyword in habit_name:
                return time
        
        return 15  # Default estimate
    
    def _estimate_success_rate(self, habit: Dict, context: Dict) -> float:
        """Estimate success rate based on habit difficulty and user context."""
        base_rate = 0.7  # 70% base success rate
        
        # Adjust based on difficulty
        difficulty_adjustment = {
            1: 0.2,   # Easy habits have higher success
            2: 0.1,
            3: 0.0,   # Medium difficulty
            4: -0.1,
            5: -0.2   # Hard habits have lower success
        }
        
        difficulty = habit.get("difficulty", 3)
        adjustment = difficulty_adjustment.get(difficulty, 0.0)
        
        # Adjust based on available time
        time_availability = context.get("time_availability", "moderate")
        if time_availability == "low" and self._estimate_habit_time(habit) > 10:
            adjustment -= 0.1
        elif time_availability == "high":
            adjustment += 0.1
        
        return max(0.3, min(0.9, base_rate + adjustment))
    
    async def _get_habit_recommendations(self, user_id: str, suggestions: List[Dict]) -> List[str]:
        """Get personalized habit formation recommendations."""
        recommendations = [
            "Start with just 1-2 habits to avoid overwhelm",
            "Focus on consistency over perfection",
            "Use habit stacking - attach new habits to existing ones",
            "Track your progress to stay motivated"
        ]
        
        # Add specific recommendations based on suggestions
        categories = set(s["category"] for s in suggestions)
        
        if "wellness" in categories:
            recommendations.append("Create a calming environment for wellness habits")
        
        if "fitness" in categories:
            recommendations.append("Prepare your workout gear the night before")
        
        if "nutrition" in categories:
            recommendations.append("Meal prep to support healthy eating habits")
        
        return recommendations
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get HabitAgent information."""
        return {
            "name": self.agent_name,
            "description": "Generates personalized habit formation recommendations",
            "capabilities": [
                "Habit suggestion generation",
                "Difficulty assessment",
                "Success rate estimation",
                "Implementation strategy",
                "Progress tracking recommendations"
            ],
            "endpoints": [
                "/api/agents/habit/suggestions",
                "/api/agents/habit/track",
                "/api/agents/habit/adjust"
            ]
        }
