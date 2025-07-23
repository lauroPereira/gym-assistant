"""
Diet Agent - Generates personalized nutrition and diet plans.
"""
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from backend.agents.base_agent import BaseMCPAgent
from backend.models.schemas import DietPlan
import logging

logger = logging.getLogger(__name__)


class DietAgent(BaseMCPAgent):
    """Agent responsible for creating and managing diet plans."""
    
    def __init__(self):
        super().__init__("DietAgent")
    
    async def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a personalized diet plan for the user."""
        try:
            # Get user profile and goals
            user_profile = await self.get_user_profile(user_id)
            
            # Get nutrition database
            nutrition_data = await self.get_resource_data("nutrition_database")
            foods = nutrition_data.get("data", []) if nutrition_data else []
            
            # Get historical meal data
            meal_history = await self.fetch_historical_data(user_id, "meal", 30)
            
            # Get nutrition goals
            nutrition_goals = [
                goal for goal in user_profile.get("goals", [])
                if goal.get("goal_type") == "nutrition"
            ]
            
            # Generate diet plan using prompt template
            diet_template = await self.get_prompt_template("diet_template")
            if diet_template:
                caloric_needs = self._calculate_caloric_needs(user_profile, context)
                formatted_prompt = self.format_prompt(diet_template, {
                    "caloric_needs": str(caloric_needs),
                    "restrictions": json.dumps(context.get("dietary_restrictions", [])),
                    "nutrition_goals": json.dumps(nutrition_goals),
                    "food_preferences": json.dumps(context.get("food_preferences", [])),
                    "activity_level": context.get("activity_level", "moderate")
                })
            
            # Create diet plan
            diet_plan = await self._create_diet_plan(
                user_id, nutrition_goals, foods, meal_history, context
            )
            
            # Log the diet plan creation
            await self.log_event(user_id, "meal", {
                "action": "diet_plan_created",
                "plan_id": diet_plan["plan_id"],
                "daily_calories": diet_plan["daily_calories"]
            })
            
            return {
                "agent": self.agent_name,
                "success": True,
                "diet_plan": diet_plan,
                "recommendations": await self._get_nutrition_recommendations(user_id, diet_plan),
                "next_actions": [
                    "Plan your meals for the week",
                    "Shop for recommended ingredients",
                    "Track your daily food intake"
                ]
            }
            
        except Exception as e:
            logger.error(f"DietAgent process error: {e}")
            return {
                "agent": self.agent_name,
                "success": False,
                "error": str(e),
                "diet_plan": None
            }
    
    def _calculate_caloric_needs(self, user_profile: Dict, context: Dict) -> int:
        """Calculate daily caloric needs based on user data."""
        # Basic calculation - in real app, use more sophisticated formulas
        base_calories = 2000  # Default
        
        # Adjust based on activity level
        activity_multipliers = {
            "sedentary": 0.8,
            "light": 0.9,
            "moderate": 1.0,
            "active": 1.2,
            "very_active": 1.4
        }
        
        activity_level = context.get("activity_level", "moderate")
        multiplier = activity_multipliers.get(activity_level, 1.0)
        
        # Adjust based on goals
        goals = user_profile.get("goals", [])
        for goal in goals:
            if "weight loss" in goal.get("title", "").lower():
                multiplier *= 0.85  # Deficit for weight loss
            elif "weight gain" in goal.get("title", "").lower():
                multiplier *= 1.15  # Surplus for weight gain
        
        return int(base_calories * multiplier)
    
    async def _create_diet_plan(
        self, 
        user_id: str, 
        goals: List[Dict], 
        foods: List[Dict], 
        history: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Create a personalized diet plan."""
        
        daily_calories = self._calculate_caloric_needs({"goals": goals}, context)
        macros = self._calculate_macros(goals, context)
        
        # Create meal structure
        meals = self._create_meal_plan(foods, daily_calories, macros, context)
        
        plan_id = f"diet_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        return {
            "plan_id": plan_id,
            "user_id": user_id,
            "daily_calories": daily_calories,
            "meals": meals,
            "macros": macros,
            "restrictions": context.get("dietary_restrictions", []),
            "duration_days": 7,  # Weekly meal plan
            "notes": "Balanced nutrition plan tailored to your goals and preferences",
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_macros(self, goals: List[Dict], context: Dict) -> Dict[str, float]:
        """Calculate macro distribution (protein, carbs, fat)."""
        # Default balanced macro distribution
        protein_percent = 25
        carbs_percent = 45
        fat_percent = 30
        
        # Adjust based on goals
        goal_text = " ".join([goal.get("title", "").lower() for goal in goals])
        
        if "muscle" in goal_text or "strength" in goal_text:
            protein_percent = 30
            carbs_percent = 40
            fat_percent = 30
        elif "endurance" in goal_text:
            protein_percent = 20
            carbs_percent = 55
            fat_percent = 25
        elif "weight loss" in goal_text:
            protein_percent = 30
            carbs_percent = 35
            fat_percent = 35
        
        return {
            "protein_percent": protein_percent,
            "carbs_percent": carbs_percent,
            "fat_percent": fat_percent
        }
    
    def _create_meal_plan(self, foods: List[Dict], daily_calories: int, macros: Dict, context: Dict) -> List[Dict[str, Any]]:
        """Create a daily meal plan."""
        restrictions = context.get("dietary_restrictions", [])
        preferences = context.get("food_preferences", [])
        
        # Filter foods based on restrictions
        available_foods = self._filter_foods(foods, restrictions, preferences)
        
        # Distribute calories across meals
        meal_distribution = {
            "breakfast": 0.25,
            "lunch": 0.30,
            "dinner": 0.35,
            "snacks": 0.10
        }
        
        meals = []
        for meal_name, calorie_ratio in meal_distribution.items():
            meal_calories = int(daily_calories * calorie_ratio)
            meal_foods = self._select_foods_for_meal(available_foods, meal_calories, meal_name)
            
            meals.append({
                "meal_type": meal_name,
                "target_calories": meal_calories,
                "foods": meal_foods,
                "total_calories": sum(food.get("calories", 0) for food in meal_foods),
                "preparation_time": self._estimate_prep_time(meal_foods)
            })
        
        return meals
    
    def _filter_foods(self, foods: List[Dict], restrictions: List[str], preferences: List[str]) -> List[Dict]:
        """Filter foods based on dietary restrictions and preferences."""
        filtered = []
        
        for food in foods:
            food_name = food.get("food_name", "").lower()
            category = food.get("category", "").lower()
            
            # Check restrictions
            skip_food = False
            for restriction in restrictions:
                restriction = restriction.lower()
                if restriction in food_name or restriction in category:
                    skip_food = True
                    break
            
            if not skip_food:
                filtered.append(food)
        
        return filtered
    
    def _select_foods_for_meal(self, foods: List[Dict], target_calories: int, meal_type: str) -> List[Dict[str, Any]]:
        """Select foods for a specific meal."""
        meal_foods = []
        current_calories = 0
        
        # Define meal-appropriate food categories
        meal_categories = {
            "breakfast": ["grain", "fruit", "dairy", "protein"],
            "lunch": ["protein", "vegetable", "grain", "fruit"],
            "dinner": ["protein", "vegetable", "grain"],
            "snacks": ["fruit", "nuts", "dairy"]
        }
        
        preferred_categories = meal_categories.get(meal_type, ["protein", "vegetable", "grain"])
        
        # Select foods from preferred categories
        for category in preferred_categories:
            category_foods = [f for f in foods if f.get("category", "").lower() == category]
            
            if category_foods and current_calories < target_calories:
                # Select a food from this category
                selected_food = category_foods[0]  # Simple selection - could be more sophisticated
                
                # Calculate portion size
                calories_per_100g = selected_food.get("calories_per_100g", 100)
                remaining_calories = min(target_calories - current_calories, target_calories * 0.4)
                portion_size = max(50, min(200, (remaining_calories / calories_per_100g) * 100))
                
                food_item = {
                    "food_name": selected_food.get("food_name"),
                    "portion_grams": round(portion_size),
                    "calories": round((calories_per_100g * portion_size) / 100),
                    "protein": round((selected_food.get("protein_per_100g", 0) * portion_size) / 100, 1),
                    "carbs": round((selected_food.get("carbs_per_100g", 0) * portion_size) / 100, 1),
                    "fat": round((selected_food.get("fat_per_100g", 0) * portion_size) / 100, 1)
                }
                
                meal_foods.append(food_item)
                current_calories += food_item["calories"]
        
        return meal_foods
    
    def _estimate_prep_time(self, foods: List[Dict]) -> int:
        """Estimate meal preparation time in minutes."""
        base_time = 10  # Base preparation time
        return base_time + len(foods) * 5  # 5 minutes per food item
    
    async def _get_nutrition_recommendations(self, user_id: str, diet_plan: Dict) -> List[str]:
        """Get personalized nutrition recommendations."""
        recommendations = [
            "Drink plenty of water throughout the day",
            "Eat regular meals to maintain energy levels",
            "Include a variety of colorful fruits and vegetables",
            "Practice portion control and mindful eating"
        ]
        
        daily_calories = diet_plan.get("daily_calories", 2000)
        
        if daily_calories < 1800:
            recommendations.extend([
                "Focus on nutrient-dense foods",
                "Consider a multivitamin supplement",
                "Monitor energy levels and adjust as needed"
            ])
        elif daily_calories > 2500:
            recommendations.extend([
                "Distribute calories across multiple meals",
                "Include healthy fats for sustained energy",
                "Time carbohydrate intake around workouts"
            ])
        
        return recommendations
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get DietAgent information."""
        return {
            "name": self.agent_name,
            "description": "Generates personalized nutrition and diet plans",
            "capabilities": [
                "Caloric needs calculation",
                "Macro distribution planning",
                "Meal plan creation",
                "Dietary restriction handling",
                "Nutrition recommendations"
            ],
            "endpoints": [
                "/api/agents/diet/plan",
                "/api/agents/diet/meals",
                "/api/agents/diet/adjust"
            ]
        }
