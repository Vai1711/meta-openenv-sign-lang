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
    previous_signs: List[str] = Field(default_factory=list)

class SignAction(BaseModel):
    action_type: ActionType
    query_sign: Optional[str] = None
    query_context: Optional[str] = None
    translation: Optional[str] = None

    def validate_action(self) -> bool:
        if self.action_type == ActionType.QUERY_DICT: return self.query_sign is not None
        if self.action_type == ActionType.QUERY_CONTEXT: return self.query_context is not None
        if self.action_type == ActionType.SUBMIT_TRANSLATION: return self.translation is not None
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
        self.signs: Dict[str, SignInfo] = self._initialize_signs()

    def _initialize_signs(self) -> Dict[str, SignInfo]:
        return {
            "APPLE": SignInfo(name="APPLE", hand_description="Fist at cheek, twist", category="food", difficulty=1),
            "WATER": SignInfo(name="WATER", hand_description="W shape, tap chin", category="drink", difficulty=2),
            "BAT": SignInfo(name="BAT", hand_description="B shape, flap wrist", audio_hint="animal", category="animal", difficulty=3),
            "BASEBALL": SignInfo(name="BASEBALL", hand_description="Swing motion", audio_hint="sports", category="sports", difficulty=3),
            "HELLO": SignInfo(name="HELLO", hand_description="Salute from forehead", category="polite", difficulty=1),
            "SQUASH_VEGETABLE": SignInfo(name="SQUASH_VEGETABLE", hand_description="Press hands", audio_hint="vegetable", category="food", difficulty=3),
            "SQUASH_SPORT": SignInfo(name="SQUASH_SPORT", hand_description="Hit alternately", audio_hint="sport", category="sports", difficulty=3)
        }

class SignInterpreterEnv:
    def __init__(self, max_steps: int = 10, seed: Optional[int] = None, task_id: Optional[int] = None):
        self.max_steps = max_steps
        self.seed = seed
        self.task_id = task_id
        self.dictionary = SignLanguageDictionary()
        self._set_difficulty(task_id)
        self.current_step = 0

    def _set_difficulty(self, task_id: Optional[int]):
        if task_id is not None:
            diffs = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
            self.current_difficulty = diffs[task_id % 3]
        else:
            self.current_difficulty = DifficultyLevel.EASY

    def reset(self, task_id: Optional[int] = None) -> SignObservation:
        self.current_step = 0
        if task_id is not None:
            self._set_difficulty(task_id)
        
        all_signs = list(self.dictionary.signs.keys())
        if self.current_difficulty == DifficultyLevel.EASY:
            self.current_sequence = [random.choice(all_signs)]
        elif self.current_difficulty == DifficultyLevel.MEDIUM:
            self.current_sequence = random.sample(all_signs, 3)
        else:
            self.current_sequence = [random.choice(["SQUASH_VEGETABLE", "SQUASH_SPORT", "BAT", "BASEBALL"])]
            
        self.target_translation = " ".join(self.current_sequence)
        return self._get_observation()

    def step(self, action: SignAction) -> Tuple[SignObservation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = 0.0
        done = False
        
        if action.action_type == ActionType.SUBMIT_TRANSLATION:
            if action.translation and action.translation.strip().upper() == self.target_translation:
                reward = 1.0 if self.current_difficulty == DifficultyLevel.EASY else 1.8
                done = True
            else:
                reward = -0.1
        
        reward -= 0.01  # Step penalty
        if self.current_step >= self.max_steps: done = True
        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> SignObservation:
        s = self.dictionary.get_sign(self.current_sequence[0])
        return SignObservation(
            hand_description=s.hand_description,
            audio_hint=s.audio_hint if self.current_difficulty == DifficultyLevel.HARD else None,
            sequence_length=len(self.current_sequence),
            current_position=0,
            difficulty=self.current_difficulty,
            previous_signs=[]
        )
    
    def state(self): return {"difficulty": self.current_difficulty}
    async def close(self): pass

def get_all_signs(): return SignLanguageDictionary().signs
