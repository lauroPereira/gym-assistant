"""
Quality of Life (QoL) Agent - Assesses and provides recommendations for overall well-being.
"""
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from backend.agents.base_agent import BaseMCPAgent
from backend.models.schemas import QoLMetrics
import logging

logger = logging.getLogger(__name__)


class QoLAgent(BaseMCPAgent):
    """Agent responsible for assessing and improving quality of life metrics."""
    
    def __init__(self):
        super().__init__("QoLAgent")
    
    async def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of life and provide improvement recommendations."""
        try:
            # Get user profile and goals
            user_profile = await self.get_user_profile(user_id)
            
            # Get various health metrics
            sleep_data = await self.fetch_historical_data(user_id, "sleep", 30)
            mood_data = await self.fetch_historical_data(user_id, "mood", 30)
            energy_data = await self.fetch_historical_data(user_id, "energy", 30)
            
            # Get QoL-related goals
            qol_goals = [
                goal for goal in user_profile.get("goals", [])
                if goal.get("goal_type") == "qol"
            ]
            
            # Generate QoL assessment using prompt template
            qol_template = await self.get_prompt_template("qol_template")
            if qol_template:
                formatted_prompt = self.format_prompt(qol_template, {
                    "sleep_data": json.dumps(sleep_data.get("data", [])),
                    "stress_indicators": json.dumps(context.get("stress_indicators", [])),
                    "social_metrics": json.dumps(context.get("social_metrics", {})),
                    "work_life_data": json.dumps(context.get("work_life_balance", {})),
                    "health_metrics": json.dumps(user_profile.get("recent_metrics", [])),
                    "mental_health_data": json.dumps(mood_data.get("data", []))
                })
            
            # Create QoL assessment
            qol_metrics = await self._assess_quality_of_life(
                user_id, user_profile, sleep_data, mood_data, energy_data, context
            )
            
            # Log the QoL assessment
            await self.log_event(user_id, "metric", {
                "action": "qol_assessment_completed",
                "overall_score": qol_metrics["overall_score"],
                "assessment_date": datetime.utcnow().isoformat()
            })
            
            return {
                "agent": self.agent_name,
                "success": True,
                "qol_metrics": qol_metrics,
                "recommendations": await self._get_qol_recommendations(user_id, qol_metrics),
                "next_actions": [
                    "Focus on the lowest scoring areas",
                    "Implement small daily improvements",
                    "Track progress weekly"
                ]
            }
            
        except Exception as e:
            logger.error(f"QoLAgent process error: {e}")
            return {
                "agent": self.agent_name,
                "success": False,
                "error": str(e),
                "qol_metrics": None
            }
    
    async def _assess_quality_of_life(
        self, 
        user_id: str, 
        user_profile: Dict, 
        sleep_data: Dict, 
        mood_data: Dict, 
        energy_data: Dict, 
        context: Dict
    ) -> Dict[str, Any]:
        """Assess overall quality of life across multiple dimensions."""
        
        # Calculate individual dimension scores
        sleep_quality = self._assess_sleep_quality(sleep_data)
        stress_level = self._assess_stress_level(context, mood_data)
        energy_level = self._assess_energy_level(energy_data)
        mood_score = self._assess_mood_score(mood_data)
        social_connections = self._assess_social_connections(context)
        work_life_balance = self._assess_work_life_balance(context)
        
        # Calculate overall score (weighted average)
        weights = {
            "sleep_quality": 0.20,
            "stress_level": 0.15,  # Lower stress = higher QoL
            "energy_level": 0.20,
            "mood_score": 0.20,
            "social_connections": 0.15,
            "work_life_balance": 0.10
        }
        
        # Invert stress level (lower stress = better QoL)
        adjusted_stress = 10.0 - stress_level
        
        overall_score = (
            sleep_quality * weights["sleep_quality"] +
            adjusted_stress * weights["stress_level"] +
            energy_level * weights["energy_level"] +
            mood_score * weights["mood_score"] +
            social_connections * weights["social_connections"] +
            work_life_balance * weights["work_life_balance"]
        )
        
        # Generate recommendations based on scores
        recommendations = self._generate_qol_recommendations({
            "sleep_quality": sleep_quality,
            "stress_level": stress_level,
            "energy_level": energy_level,
            "mood_score": mood_score,
            "social_connections": social_connections,
            "work_life_balance": work_life_balance
        })
        
        return {
            "user_id": user_id,
            "overall_score": round(overall_score, 1),
            "sleep_quality": round(sleep_quality, 1),
            "stress_level": round(stress_level, 1),
            "energy_level": round(energy_level, 1),
            "mood_score": round(mood_score, 1),
            "social_connections": round(social_connections, 1),
            "work_life_balance": round(work_life_balance, 1),
            "recommendations": recommendations,
            "assessment_date": datetime.utcnow().isoformat()
        }
    
    def _assess_sleep_quality(self, sleep_data: Dict) -> float:
        """Assess sleep quality based on historical data."""
        data_points = sleep_data.get("data", [])
        
        if not data_points:
            return 5.0  # Neutral score if no data
        
        # Calculate average sleep duration and quality
        total_hours = 0
        quality_sum = 0
        count = 0
        
        for point in data_points:
            event_data = point.get("event_data", {})
            if "sleep_hours" in event_data:
                hours = event_data["sleep_hours"]
                quality = event_data.get("sleep_quality", 5)  # 1-10 scale
                
                total_hours += hours
                quality_sum += quality
                count += 1
        
        if count == 0:
            return 5.0
        
        avg_hours = total_hours / count
        avg_quality = quality_sum / count
        
        # Score based on optimal sleep (7-9 hours) and quality
        hours_score = 10.0 if 7 <= avg_hours <= 9 else max(0, 10 - abs(avg_hours - 8))
        quality_score = avg_quality
        
        return (hours_score + quality_score) / 2
    
    def _assess_stress_level(self, context: Dict, mood_data: Dict) -> float:
        """Assess stress level based on indicators and mood data."""
        stress_indicators = context.get("stress_indicators", [])
        
        # Base stress level
        base_stress = 5.0
        
        # Adjust based on stress indicators
        stress_factors = {
            "work_pressure": 2.0,
            "financial_concerns": 1.5,
            "relationship_issues": 1.5,
            "health_concerns": 2.0,
            "sleep_problems": 1.0,
            "time_pressure": 1.0
        }
        
        for indicator in stress_indicators:
            if indicator in stress_factors:
                base_stress += stress_factors[indicator]
        
        # Adjust based on mood data
        mood_points = mood_data.get("data", [])
        if mood_points:
            recent_moods = [p.get("value", 5) for p in mood_points[-7:]]  # Last week
            avg_mood = sum(recent_moods) / len(recent_moods)
            
            # Lower mood correlates with higher stress
            mood_stress_adjustment = (5 - avg_mood) * 0.5
            base_stress += mood_stress_adjustment
        
        return min(10.0, max(1.0, base_stress))
    
    def _assess_energy_level(self, energy_data: Dict) -> float:
        """Assess energy level based on historical data."""
        data_points = energy_data.get("data", [])
        
        if not data_points:
            return 5.0  # Neutral score if no data
        
        # Calculate average energy level from recent data
        recent_energy = [p.get("value", 5) for p in data_points[-14:]]  # Last 2 weeks
        
        if not recent_energy:
            return 5.0
        
        return sum(recent_energy) / len(recent_energy)
    
    def _assess_mood_score(self, mood_data: Dict) -> float:
        """Assess mood score based on historical data."""
        data_points = mood_data.get("data", [])
        
        if not data_points:
            return 5.0  # Neutral score if no data
        
        # Calculate average mood from recent data
        recent_moods = [p.get("value", 5) for p in data_points[-14:]]  # Last 2 weeks
        
        if not recent_moods:
            return 5.0
        
        return sum(recent_moods) / len(recent_moods)
    
    def _assess_social_connections(self, context: Dict) -> float:
        """Assess social connections quality."""
        social_metrics = context.get("social_metrics", {})
        
        # Default score
        score = 5.0
        
        # Adjust based on social indicators
        close_friends = social_metrics.get("close_friends", 0)
        social_activities = social_metrics.get("weekly_social_activities", 0)
        family_contact = social_metrics.get("family_contact_frequency", 1)  # 1-7 scale
        
        # Score based on social connections
        if close_friends >= 3:
            score += 1.0
        elif close_friends >= 1:
            score += 0.5
        else:
            score -= 1.0
        
        if social_activities >= 2:
            score += 1.0
        elif social_activities >= 1:
            score += 0.5
        else:
            score -= 0.5
        
        score += (family_contact - 3) * 0.3  # Adjust based on family contact
        
        return min(10.0, max(1.0, score))
    
    def _assess_work_life_balance(self, context: Dict) -> float:
        """Assess work-life balance."""
        work_life_data = context.get("work_life_balance", {})
        
        # Default score
        score = 5.0
        
        # Adjust based on work-life indicators
        work_hours = work_life_data.get("weekly_work_hours", 40)
        overtime_frequency = work_life_data.get("overtime_frequency", 1)  # 1-5 scale
        vacation_days = work_life_data.get("annual_vacation_days", 14)
        work_satisfaction = work_life_data.get("work_satisfaction", 5)  # 1-10 scale
        
        # Adjust based on work hours
        if work_hours <= 40:
            score += 1.0
        elif work_hours <= 50:
            score += 0.0
        else:
            score -= (work_hours - 50) * 0.1
        
        # Adjust based on overtime
        score -= (overtime_frequency - 1) * 0.5
        
        # Adjust based on vacation
        if vacation_days >= 20:
            score += 1.0
        elif vacation_days >= 10:
            score += 0.5
        else:
            score -= 0.5
        
        # Adjust based on work satisfaction
        score += (work_satisfaction - 5) * 0.3
        
        return min(10.0, max(1.0, score))
    
    def _generate_qol_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on QoL scores."""
        recommendations = []
        
        # Sleep recommendations
        if scores["sleep_quality"] < 6:
            recommendations.extend([
                "Establish a consistent sleep schedule",
                "Create a relaxing bedtime routine",
                "Limit screen time before bed"
            ])
        
        # Stress recommendations
        if scores["stress_level"] > 7:
            recommendations.extend([
                "Practice daily stress management techniques",
                "Consider meditation or mindfulness",
                "Identify and address stress triggers"
            ])
        
        # Energy recommendations
        if scores["energy_level"] < 6:
            recommendations.extend([
                "Review your nutrition and hydration",
                "Incorporate regular physical activity",
                "Check for underlying health issues"
            ])
        
        # Mood recommendations
        if scores["mood_score"] < 6:
            recommendations.extend([
                "Engage in activities you enjoy",
                "Consider talking to a mental health professional",
                "Practice gratitude and positive thinking"
            ])
        
        # Social recommendations
        if scores["social_connections"] < 6:
            recommendations.extend([
                "Reach out to friends and family regularly",
                "Join social groups or activities",
                "Consider volunteering or community involvement"
            ])
        
        # Work-life balance recommendations
        if scores["work_life_balance"] < 6:
            recommendations.extend([
                "Set clear boundaries between work and personal time",
                "Take regular breaks during work",
                "Plan and take vacation time"
            ])
        
        return recommendations
    
    async def _get_qol_recommendations(self, user_id: str, qol_metrics: Dict) -> List[str]:
        """Get personalized quality of life recommendations."""
        recommendations = qol_metrics.get("recommendations", [])
        
        # Add general QoL recommendations
        general_recommendations = [
            "Focus on progress, not perfection",
            "Make small, sustainable changes",
            "Celebrate your improvements",
            "Seek support when needed"
        ]
        
        recommendations.extend(general_recommendations)
        
        # Prioritize based on lowest scores
        scores = {
            "sleep": qol_metrics.get("sleep_quality", 5),
            "stress": 10 - qol_metrics.get("stress_level", 5),  # Invert for priority
            "energy": qol_metrics.get("energy_level", 5),
            "mood": qol_metrics.get("mood_score", 5),
            "social": qol_metrics.get("social_connections", 5),
            "work_life": qol_metrics.get("work_life_balance", 5)
        }
        
        # Find the area that needs most improvement
        lowest_area = min(scores, key=scores.get)
        
        priority_recommendations = {
            "sleep": "Prioritize improving your sleep quality and consistency",
            "stress": "Focus on stress reduction and management techniques",
            "energy": "Work on boosting your energy through lifestyle changes",
            "mood": "Consider activities and practices that improve your mood",
            "social": "Invest time in building and maintaining social connections",
            "work_life": "Create better boundaries between work and personal life"
        }
        
        if lowest_area in priority_recommendations:
            recommendations.insert(0, priority_recommendations[lowest_area])
        
        return recommendations[:10]  # Limit to 10 recommendations
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get QoLAgent information."""
        return {
            "name": self.agent_name,
            "description": "Assesses and provides recommendations for overall quality of life",
            "capabilities": [
                "Multi-dimensional QoL assessment",
                "Sleep quality analysis",
                "Stress level evaluation",
                "Mood and energy tracking",
                "Social connection assessment",
                "Work-life balance evaluation"
            ],
            "endpoints": [
                "/api/agents/qol/assessment",
                "/api/agents/qol/metrics",
                "/api/agents/qol/recommendations"
            ]
        }
