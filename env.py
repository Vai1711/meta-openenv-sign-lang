"""
Sign Language Interpreter Environment for Meta PyTorch OpenEnv Hackathon
Complete environment implementation with Pydantic models and task logic
"""

import random
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import json
# from openenv.core import SupportEnv  # Commented for testing


class ActionType(str, Enum):
    """Action types for the Sign Language Interpreter"""
    QUERY_DICT = "query_dict"
    QUERY_CONTEXT = "query_context"
    SUBMIT_TRANSLATION = "submit_translation"


class DifficultyLevel(str, Enum):
    """Difficulty levels for tasks"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class SignObservation(BaseModel):
    """
    Observation model representing what the agent sees in the environment
    """
    hand_description: str = Field(..., description="Visual description of hand movements and positions")
    facial_expression: Optional[str] = Field(None, description="Facial expression accompanying the sign")
    audio_hint: Optional[str] = Field(None, description="Audio hint to disambiguate similar signs")
    context_clue: Optional[str] = Field(None, description="Context clue for disambiguation")
    sequence_length: int = Field(1, description="Number of signs in the current sequence")
    current_position: int = Field(0, description="Current position in the sequence")
    difficulty: DifficultyLevel = Field(..., description="Current task difficulty")
    previous_signs: List[str] = Field(default_factory=list, description="Previously observed signs in sequence")
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        }


class SignAction(BaseModel):
    """
    Action model for agent interactions with the environment
    """
    action_type: ActionType = Field(..., description="Type of action to perform")
    query_sign: Optional[str] = Field(None, description="Sign to query in dictionary (for query_dict)")
    query_context: Optional[str] = Field(None, description="Sign to query context for (for query_context)")
    translation: Optional[str] = Field(None, description="Translation answer (for submit_translation)")
    
    def validate_action(self) -> bool:
        """Validate that the action has the required fields for its type"""
        if self.action_type == ActionType.QUERY_DICT:
            return self.query_sign is not None
        elif self.action_type == ActionType.QUERY_CONTEXT:
            return self.query_context is not None
        elif self.action_type == ActionType.SUBMIT_TRANSLATION:
            return self.translation is not None
        return False


class SignInfo(BaseModel):
    """Information about a single ASL sign"""
    name: str = Field(..., description="Name of the sign")
    hand_description: str = Field(..., description="How to perform the sign with hands")
    facial_expression: Optional[str] = Field(None, description="Facial expression context")
    audio_hint: Optional[str] = Field(None, description="Audio hint for disambiguation")
    context_clue: Optional[str] = Field(None, description="Context clue for disambiguation")
    category: str = Field(..., description="Category of the sign (food, animal, etc.)")
    difficulty: int = Field(1, ge=1, le=5, description="Difficulty level (1-5)")


class SignLanguageDictionary:
    """Dictionary of ASL signs with detailed descriptions"""
    
    def __init__(self):
        self.signs: Dict[str, SignInfo] = self._initialize_signs()
    
    def _initialize_signs(self) -> Dict[str, SignInfo]:
        """Initialize hardcoded dictionary of 15 ASL signs"""
        return {
            "APPLE": SignInfo(
                name="APPLE",
                hand_description="Place fist at cheek and twist as if turning an apple stem",
                facial_expression="Neutral",
                audio_hint=None,
                category="food",
                difficulty=1
            ),
            "BANANA": SignInfo(
                name="BANANA",
                hand_description="Extend index finger and trace banana shape downward",
                facial_expression="Neutral",
                audio_hint=None,
                category="food",
                difficulty=1
            ),
            "WATER": SignInfo(
                name="WATER",
                hand_description="Form W shape with fingers and tap chin repeatedly",
                facial_expression="Neutral",
                audio_hint=None,
                category="drink",
                difficulty=2
            ),
            "KNOW": SignInfo(
                name="KNOW",
                hand_description="Place fingertips on forehead then move forward",
                facial_expression="Neutral",
                audio_hint=None,
                category="verb",
                difficulty=2
            ),
            "ME": SignInfo(
                name="ME",
                hand_description="Point index finger to chest",
                facial_expression="Neutral",
                audio_hint=None,
                category="pronoun",
                difficulty=1
            ),
            "WANT": SignInfo(
                name="WANT",
                hand_description="Extend hands forward with palms up and pull back",
                facial_expression="Slightly pleading",
                audio_hint=None,
                category="verb",
                difficulty=2
            ),
            "BAT": SignInfo(
                name="BAT",
                hand_description="Form B shape with thumb and fingers, flap at wrist",
                facial_expression="Neutral",
                audio_hint="animal",
                category="animal",
                difficulty=3
            ),
            "BASEBALL": SignInfo(
                name="BASEBALL",
                hand_description="Form fists and swing as if holding a bat",
                facial_expression="Focused",
                audio_hint="sports equipment",
                category="sports",
                difficulty=3
            ),
            "CAT": SignInfo(
                name="CAT",
                hand_description="Pinch cheeks with thumb and index finger, pull outward",
                facial_expression="Neutral",
                audio_hint=None,
                category="animal",
                difficulty=2
            ),
            "DOG": SignInfo(
                name="DOG",
                hand_description="Pat leg and snap fingers",
                facial_expression="Friendly",
                audio_hint=None,
                category="animal",
                difficulty=2
            ),
            "BOOK": SignInfo(
                name="BOOK",
                hand_description="Open hands together like opening a book",
                facial_expression="Neutral",
                audio_hint=None,
                category="object",
                difficulty=1
            ),
            "READ": SignInfo(
                name="READ",
                hand_description="Form open book shape with hands and move as if reading",
                facial_expression="Focused",
                audio_hint=None,
                category="verb",
                difficulty=2
            ),
            "HELP": SignInfo(
                name="HELP",
                hand_description="Lift fist upward with palm facing up",
                facial_expression="Concerned",
                audio_hint=None,
                category="verb",
                difficulty=2
            ),
            "THANK": SignInfo(
                name="THANK",
                hand_description="Move flat hand from chin downward and forward",
                facial_expression="Sincere",
                audio_hint=None,
                category="polite",
                difficulty=1
            ),
            "PLEASE": SignInfo(
                name="PLEASE",
                hand_description="Make circular motion on chest with flat hand",
                facial_expression="Polite",
                audio_hint=None,
                category="polite",
                difficulty=1
            ),
            "HOUSE": SignInfo(
                name="HOUSE",
                hand_description="Form flat hands, draw roof shape then walls",
                facial_expression="Neutral",
                audio_hint=None,
                category="location",
                difficulty=2
            ),
            "CAR": SignInfo(
                name="CAR",
                hand_description="Form fists and move as if steering a wheel",
                facial_expression="Neutral",
                audio_hint=None,
                category="transportation",
                difficulty=1
            ),
            "MILK": SignInfo(
                name="MILK",
                hand_description="Squeeze hand as if milking a cow, repeated motion",
                facial_expression="Neutral",
                audio_hint=None,
                category="drink",
                difficulty=2
            ),
            "BREAD": SignInfo(
                name="BREAD",
                hand_description="Slice bread with flat hand motion",
                facial_expression="Neutral",
                audio_hint=None,
                category="food",
                difficulty=2
            ),
            "FISH": SignInfo(
                name="FISH",
                hand_description="Move flat hand forward like swimming fish",
                facial_expression="Neutral",
                audio_hint=None,
                category="animal",
                difficulty=2
            ),
            "BIRD": SignInfo(
                name="BIRD",
                hand_description="Open and close fingers and thumb like beak, flap at wrist",
                facial_expression="Neutral",
                audio_hint=None,
                category="animal",
                difficulty=3
            ),
            "TREE": SignInfo(
                name="TREE",
                hand_description="Raise forearm as trunk, spread fingers as branches",
                facial_expression="Neutral",
                audio_hint=None,
                category="nature",
                difficulty=2
            ),
            "FLOWER": SignInfo(
                name="FLOWER",
                hand_description="Form bud with fist, then open fingers like blooming",
                facial_expression="Gentle",
                audio_hint=None,
                category="nature",
                difficulty=3
            ),
            "COMPUTER": SignInfo(
                name="COMPUTER",
                hand_description="Type with both hands as if on keyboard",
                facial_expression="Focused",
                audio_hint=None,
                category="technology",
                difficulty=2
            ),
            "PHONE": SignInfo(
                name="PHONE",
                hand_description="Form Y shape with thumb and pinky, hold to ear",
                facial_expression="Neutral",
                audio_hint=None,
                category="technology",
                difficulty=1
            ),
            "MUSIC": SignInfo(
                name="MUSIC",
                hand_description="Wave arms like conducting orchestra",
                facial_expression="Expressive",
                audio_hint=None,
                category="art",
                difficulty=2
            ),
            "DANCE": SignInfo(
                name="DANCE",
                hand_description="Swing arms and body in rhythmic motion",
                facial_expression="Happy",
                audio_hint=None,
                category="art",
                difficulty=2
            ),
            "SLEEP": SignInfo(
                name="SLEEP",
                hand_description="Rest cheek on flat hands, close eyes",
                facial_expression="Peaceful",
                audio_hint=None,
                category="action",
                difficulty=1
            ),
            "WORK": SignInfo(
                name="WORK",
                hand_description="Tap fists together repeatedly",
                facial_expression="Determined",
                audio_hint=None,
                category="action",
                difficulty=2
            ),
            "SQUASH_VEGETABLE": SignInfo(
                name="SQUASH_VEGETABLE",
                hand_description="Form C shapes with both hands and press together as if holding a vegetable",
                facial_expression="Neutral",
                audio_hint="vegetable",
                category="food",
                difficulty=3
            ),
            "SQUASH_SPORT": SignInfo(
                name="SQUASH_SPORT",
                hand_description="Form C shapes with both hands and hit alternately as if playing racquet sport",
                facial_expression="Focused",
                audio_hint="sport",
                context_clue="The user is at a sports club",
                category="sports",
                difficulty=3
            ),
            "ORANGE_FRUIT": SignInfo(
                name="ORANGE_FRUIT",
                hand_description="Form O shape with fingers and squeeze as if holding an orange fruit",
                facial_expression="Neutral",
                audio_hint="fruit",
                context_clue="The user is at a grocery store",
                category="food",
                difficulty=3
            ),
            "ORANGE_COLOR": SignInfo(
                name="ORANGE_COLOR",
                hand_description="Form O shape with fingers and move in front of body as if showing orange color",
                facial_expression="Neutral",
                audio_hint="color",
                context_clue="The user is describing colors in a painting",
                category="color",
                difficulty=3
            )
        }
    
    def get_sign(self, sign_name: str) -> Optional[SignInfo]:
        """Get sign information by name"""
        return self.signs.get(sign_name.upper())
    
    def get_all_signs(self) -> Dict[str, SignInfo]:
        """Get all signs"""
        return self.signs.copy()
    
    def get_signs_by_category(self, category: str) -> Dict[str, SignInfo]:
        """Get signs filtered by category"""
        return {k: v for k, v in self.signs.items() if v.category == category}
    
    def search_signs(self, query: str) -> List[str]:
        """Search for signs by name or description"""
        query = query.lower()
        results = []
        for name, sign_info in self.signs.items():
            if (query in name.lower() or 
                query in sign_info.hand_description.lower() or
                (sign_info.audio_hint and query in sign_info.audio_hint.lower())):
                results.append(name)
        return results


# class SignInterpreterEnv(SupportEnv):  # Commented for testing
class SignInterpreterEnv:
    """
    Main environment class for Sign Language Interpreter
    Implements openenv-core structure with reset(), step(), and state() methods
    """
    
    def __init__(self, max_steps: int = 10, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.dictionary = SignLanguageDictionary()
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0
        
        # Episode state
        self.current_sequence: List[str] = []
        self.current_position = 0
        self.current_difficulty = DifficultyLevel.EASY
        self.target_translation: str = ""
        self.episode_reward = 0.0
        self.query_count = 0
        self.correct_queries = 0
        
        # Statistics
        self.stats = {
            "total_episodes": 0,
            "total_reward": 0.0,
            "success_rate": 0.0,
            "average_steps_per_episode": 0.0
        }
    
    def reset(self) -> SignObservation:
        """Reset the environment for a new episode"""
        self.current_episode += 1
        self.current_step = 0
        self.episode_reward = 0.0
        self.query_count = 0
        self.correct_queries = 0
        
        # Randomly select difficulty
        self.current_difficulty = random.choice(list(DifficultyLevel))
        
        # Generate sequence based on difficulty
        self.current_sequence = self._generate_sequence()
        self.current_position = 0
        self.target_translation = " ".join(self.current_sequence)
        
        return self._get_observation()
    
    def step(self, action: SignAction) -> Tuple[SignObservation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, done, info
        """
        if not action.validate_action():
            raise ValueError(f"Invalid action: {action}")
        
        self.current_step += 1
        self.total_steps += 1
        
        reward = 0.0
        done = False
        info = {}
        
        if action.action_type == ActionType.QUERY_DICT:
            # Handle dictionary query
            self.query_count += 1
            sign_info = self.dictionary.get_sign(action.query_sign)
            
            if sign_info:
                # Cap query rewards to prevent exploitation
                if self.query_count <= len(self.current_sequence):
                    reward = 0.2  # +0.2 for correct query
                    self.correct_queries += 1
                else:
                    reward = 0.0  # No reward for excessive queries
                info["query_result"] = sign_info.model_dump()
            else:
                reward = -0.1  # -0.1 for wrong query
                info["query_result"] = None
            
            info["action_type"] = "query_dict"
            
        elif action.action_type == ActionType.QUERY_CONTEXT:
            # Handle context query
            self.query_count += 1
            sign_info = self.dictionary.get_sign(action.query_context)
            
            if sign_info:
                # Cap query rewards to prevent exploitation
                if self.query_count <= len(self.current_sequence):
                    reward = 0.2  # +0.2 for correct context query
                    self.correct_queries += 1
                else:
                    reward = 0.0  # No reward for excessive queries
                info["context_result"] = sign_info.context_clue
            else:
                reward = -0.1  # -0.1 for wrong context query
                info["context_result"] = None
            
            info["action_type"] = "query_context"
        elif action.action_type == ActionType.SUBMIT_TRANSLATION:
            # Handle translation submission
            if action.translation and action.translation.strip().lower() == self.target_translation.strip().lower():
                # Reward scaling based on difficulty
                if self.current_difficulty == DifficultyLevel.EASY:
                    reward = 1.0
                elif self.current_difficulty == DifficultyLevel.MEDIUM:
                    reward = 1.5
                else:  # HARD
                    reward = 2.0
                info["correct"] = True
                done = True
            else:
                reward = -0.1  # -0.1 for wrong translation
                info["correct"] = False
                # Don't end episode on wrong answer, let agent try again
            
            info["action_type"] = "submit_translation"
            info["target_translation"] = self.target_translation
            info["submitted_translation"] = action.translation
        
        self.episode_reward += reward
        
        # Add small step penalty to encourage efficiency
        reward -= 0.01
        
        # Check episode termination conditions
        if self.current_step >= self.max_steps:
            done = True
            info["termination_reason"] = "max_steps_reached"
        
        # Update statistics
        self._update_stats(done)
        
        observation = self._get_observation()
        return observation, reward, done, info
    
    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment
        Required for openenv-core compliance
        """
        return {
            "episode": self.current_episode,
            "step": self.current_step,
            "total_steps": self.total_steps,
            "current_sequence": self.current_sequence.copy(),
            "current_position": self.current_position,
            "target_translation": self.target_translation,
            "difficulty": self.current_difficulty.value,
            "episode_reward": self.episode_reward,
            "query_count": self.query_count,
            "correct_queries": self.correct_queries,
            "stats": self.stats.copy()
        }
    
    def _generate_sequence(self) -> List[str]:
        """Generate a sign sequence based on difficulty level"""
        all_signs = list(self.dictionary.signs.keys())
        
        if self.current_difficulty == DifficultyLevel.EASY:
            # Single sign
            return [random.choice(all_signs)]
        
        elif self.current_difficulty == DifficultyLevel.MEDIUM:
            # 3-word sequence
            return random.sample(all_signs, min(3, len(all_signs)))
        
        else:  # HARD
            # Use audio hint to distinguish similar signs
            # Focus on BAT vs BASEBALL or other similar pairs
            similar_pairs = [
                (["BAT"], "animal"),
                (["BASEBALL"], "sports equipment"),
                (["CAT", "DOG"], "animal"),
                (["BOOK", "READ"], "object/verb"),
                (["SQUASH_VEGETABLE"], "vegetable"),
                (["SQUASH_SPORT"], "sport"),
                (["ORANGE_FRUIT"], "fruit"),
                (["ORANGE_COLOR"], "color")
            ]
            
            pair, hint = random.choice(similar_pairs)
            if isinstance(pair, list) and len(pair) > 1:
                # Choose one from the pair
                selected = random.choice(pair)
            else:
                selected = pair[0] if isinstance(pair, list) else pair
            
            return [selected]
    
    def _get_observation(self) -> SignObservation:
        """Get current observation based on sequence position"""
        if self.current_position < len(self.current_sequence):
            current_sign = self.current_sequence[self.current_position]
            sign_info = self.dictionary.get_sign(current_sign)
            
            if sign_info:
                # For hard difficulty, include audio hint and context clue for disambiguation
                audio_hint = sign_info.audio_hint if self.current_difficulty == DifficultyLevel.HARD else None
                context_clue = sign_info.context_clue if self.current_difficulty == DifficultyLevel.HARD else None
                
                return SignObservation(
                    hand_description=sign_info.hand_description,
                    facial_expression=sign_info.facial_expression,
                    audio_hint=audio_hint,
                    context_clue=context_clue,
                    sequence_length=len(self.current_sequence),
                    current_position=self.current_position,
                    difficulty=self.current_difficulty,
                    previous_signs=self.current_sequence[:self.current_position]
                )
        
        # Fallback observation
        return SignObservation(
            hand_description="No sign currently visible",
            context_clue=None,
            sequence_length=len(self.current_sequence),
            current_position=self.current_position,
            difficulty=self.current_difficulty,
            previous_signs=self.current_sequence[:self.current_position]
        )
    
    def _update_stats(self, episode_done: bool):
        """Update environment statistics"""
        if episode_done:
            self.stats["total_episodes"] += 1
            self.stats["total_reward"] += self.episode_reward
            
            # Calculate success rate (episodes with positive reward)
            if self.stats["total_episodes"] > 0:
                success = 1.0 if self.episode_reward > 0 else 0.0
                total_success = self.stats["success_rate"] * (self.stats["total_episodes"] - 1) + success
                self.stats["success_rate"] = total_success / self.stats["total_episodes"]
                
                self.stats["average_steps_per_episode"] = (
                    self.stats["average_steps_per_episode"] * (self.stats["total_episodes"] - 1) + 
                    self.current_step
                ) / self.stats["total_episodes"]
    
    def get_dictionary_info(self) -> Dict[str, Any]:
        """Get information about the sign dictionary"""
        return {
            "total_signs": len(self.dictionary.signs),
            "categories": list(set(sign.category for sign in self.dictionary.signs.values())),
            "difficulty_distribution": {
                difficulty: len([s for s in self.dictionary.signs.values() if s.difficulty == difficulty])
                for difficulty in range(1, 6)
            }
        }
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment state"""
        if mode == "human":
            print(f"Episode: {self.current_episode}, Step: {self.current_step}")
            print(f"Difficulty: {self.current_difficulty.value}")
            print(f"Current Sequence: {' -> '.join(self.current_sequence)}")
            print(f"Target Translation: {self.target_translation}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("-" * 50)
        elif mode == "string":
            return f"Episode {self.current_episode}, Step {self.current_step}, Difficulty: {self.current_difficulty.value}, Target: {self.target_translation}"
        
        return None


# Global environment instance for easy access
ENV_CONFIG = SignLanguageEnv = SignInterpreterEnv


# Utility functions for easier usage
def create_env(difficulty: Optional[str] = None, max_steps: int = 10, seed: Optional[int] = None) -> SignInterpreterEnv:
    """Create a new environment instance"""
    env = SignInterpreterEnv(max_steps=max_steps, seed=seed)
    if difficulty:
        env.current_difficulty = DifficultyLevel(difficulty)
    return env


def get_all_signs() -> Dict[str, SignInfo]:
    """Get all available signs"""
    return SignLanguageDictionary().get_all_signs()


def search_signs(query: str) -> List[str]:
    """Search for signs by name or description"""
    return SignLanguageDictionary().search_signs(query)
