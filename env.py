import random
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum

class ActionType(str, Enum):
    QUERY_DICT = "query_dict"
    QUERY_CONTEXT = "query_context"
    SUBMIT_TRANSLATION = "submit_translation"

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class SignObservation(BaseModel):
    hand_description: str
    facial_expression: Optional[str] = None
    audio_hint: Optional[str] = None
    context_clue: Optional[str] = None
    sequence_length: int = 1
    current_position: int = 0
    difficulty: DifficultyLevel
    previous_signs: List[str] = []

class SignAction(BaseModel):
    action_type: ActionType
    query_sign: Optional[str] = None
    query_context: Optional[str] = None
    translation: Optional[str] = None

    def validate_action(self) -> bool:
        if self.action_type == ActionType.QUERY_DICT:
            return self.query_sign is not None
        if self.action_type == ActionType.QUERY_CONTEXT:
            return self.query_context is not None
        if self.action_type == ActionType.SUBMIT_TRANSLATION:
            return self.translation is not None
        return False

class SignInfo(BaseModel):
    name: str
    hand_description: str
    facial_expression: Optional[str] = None
    audio_hint: Optional[str] = None
    context_clue: Optional[str] = None
    category: str
    difficulty: int

class SignLanguageDictionary:
    def __init__(self):
        self.signs = self._initialize_signs()

    def _initialize_signs(self) -> Dict[str, SignInfo]:
        # Keeping your core 33 signs (Truncated here for brevity, keep your full list)
        return {
            "APPLE": SignInfo(name="APPLE", hand_description="Twist fist at cheek", category="food", difficulty=1),
            "BAT": SignInfo(name="BAT", hand_description="B-shape flap at wrist", audio_hint="animal", category="animal", difficulty=3),
            "BASEBALL": SignInfo(name="BASEBALL", hand_description="Swing fists like a bat", audio_hint="sports equipment", category="sports", difficulty=3),
            "SQUASH_VEGETABLE": SignInfo(name="SQUASH_VEGETABLE", hand_description="Press C-shapes", audio_hint="vegetable", category="food", difficulty=3),
            "SQUASH_SPORT": SignInfo(name="SQUASH_SPORT", hand_description="Hit C-shapes alternately", audio_hint="sport", context_clue="At a sports club", category="sports", difficulty=3),
            # ... Include your other signs here ...
        }

    def get_sign(self, name: str) -> Optional[SignInfo]:
        return self.signs.get(name.upper())

class SignInterpreterEnv:
    def __init__(self, max_steps: int = 10, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.dictionary = SignLanguageDictionary()
        if seed: random.seed(seed)

    def reset(self, task_id: Optional[int] = None) -> SignObservation:
        self.current_step = 0
        self.episode_reward = 0.0
        
        # FIX: Explicit Task ID mapping for the grader
        if task_id == 0: self.current_difficulty = DifficultyLevel.EASY
        elif task_id == 1: self.current_difficulty = DifficultyLevel.MEDIUM
        elif task_id == 2: self.current_difficulty = DifficultyLevel.HARD
        else: self.current_difficulty = DifficultyLevel.EASY
        
        self.current_sequence = self._generate_sequence()
        self.target_translation = " ".join(self.current_sequence)
        return self._get_observation()

    def _generate_sequence(self) -> List[str]:
        all_signs = list(self.dictionary.signs.keys())
        if self.current_difficulty == DifficultyLevel.EASY:
            return [random.choice(all_signs)]
        elif self.current_difficulty == DifficultyLevel.MEDIUM:
            return random.sample(all_signs, 3)
        else: # HARD
            # Logic to force ambiguous pairs
            pairs = [["BAT", "BASEBALL"], ["SQUASH_VEGETABLE", "SQUASH_SPORT"]]
            return [random.choice(random.choice(pairs))]

    def step(self, action: SignAction) -> Tuple[SignObservation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = -0.01 # Step penalty
        done = False
        
        if action.action_type == ActionType.SUBMIT_TRANSLATION:
            if action.translation and action.translation.strip().upper() == self.target_translation:
                reward = {"easy": 1.0, "medium": 1.5, "hard": 2.0}[self.current_difficulty.value]
                done = True
            else:
                reward -= 0.1
        
        if self.current_step >= self.max_steps: done = True
        self.episode_reward += reward
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> SignObservation:
        sign = self.dictionary.get_sign(self.current_sequence[0])
        return SignObservation(
            hand_description=sign.hand_description,
            facial_expression=sign.facial_expression,
            audio_hint=sign.audio_hint if self.current_difficulty == DifficultyLevel.HARD else None,
            context_clue=sign.context_clue if self.current_difficulty == DifficultyLevel.HARD else None,
            difficulty=self.current_difficulty,
            sequence_length=len(self.current_sequence)
        )

def get_all_signs(): return SignLanguageDictionary().signs
