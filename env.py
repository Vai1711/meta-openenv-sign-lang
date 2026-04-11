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
        # FULL DICTIONARY (33 SIGNS)
        return {
            "APPLE": SignInfo(name="APPLE", hand_description="Twist fist at cheek", category="food", difficulty=1),
            "BANANA": SignInfo(name="BANANA", hand_description="Trace banana shape downward", category="food", difficulty=1),
            "WATER": SignInfo(name="WATER", hand_description="W shape tap chin", category="drink", difficulty=2),
            "ME": SignInfo(name="ME", hand_description="Point to chest", category="pronoun", difficulty=1),
            "WANT": SignInfo(name="WANT", hand_description="Pull hands back palms up", category="verb", difficulty=2),
            "BAT": SignInfo(name="BAT", hand_description="B-shape flap at wrist", audio_hint="animal", category="animal", difficulty=3),
            "BASEBALL": SignInfo(name="BASEBALL", hand_description="Swing fists like a bat", audio_hint="sports equipment", category="sports", difficulty=3),
            "CAT": SignInfo(name="CAT", hand_description="Pinch cheeks pull out", category="animal", difficulty=2),
            "DOG": SignInfo(name="DOG", hand_description="Pat leg snap fingers", category="animal", difficulty=2),
            "BOOK": SignInfo(name="BOOK", hand_description="Open hands like a book", category="object", difficulty=1),
            "READ": SignInfo(name="READ", hand_description="Move eyes over open hands", category="verb", difficulty=2),
            "HELP": SignInfo(name="HELP", hand_description="Lift fist with flat hand", category="verb", difficulty=2),
            "THANK": SignInfo(name="THANK", hand_description="Hand from chin forward", category="polite", difficulty=1),
            "PLEASE": SignInfo(name="PLEASE", hand_description="Circle on chest", category="polite", difficulty=1),
            "HOUSE": SignInfo(name="HOUSE", hand_description="Draw roof and walls", category="location", difficulty=2),
            "CAR": SignInfo(name="CAR", hand_description="Steer imaginary wheel", category="transportation", difficulty=1),
            "MILK": SignInfo(name="MILK", hand_description="Squeeze repeated motion", category="drink", difficulty=2),
            "BREAD": SignInfo(name="BREAD", hand_description="Slice flat hand", category="food", difficulty=2),
            "FISH": SignInfo(name="FISH", hand_description="Wiggle hand like swimming", category="animal", difficulty=2),
            "BIRD": SignInfo(name="BIRD", hand_description="Beak shape at mouth", category="animal", difficulty=3),
            "TREE": SignInfo(name="TREE", hand_description="Arm as trunk shake hand", category="nature", difficulty=2),
            "FLOWER": SignInfo(name="FLOWER", hand_description="Bud to bloom fingers", category="nature", difficulty=3),
            "COMPUTER": SignInfo(name="COMPUTER", hand_description="Type on keyboard", category="technology", difficulty=2),
            "PHONE": SignInfo(name="PHONE", hand_description="Y shape to ear", category="technology", difficulty=1),
            "MUSIC": SignInfo(name="MUSIC", hand_description="Wave conducting orchestra", category="art", difficulty=2),
            "DANCE": SignInfo(name="DANCE", hand_description="Swing body rhythmically", category="art", difficulty=2),
            "SLEEP": SignInfo(name="SLEEP", hand_description="Cheek on hands", category="action", difficulty=1),
            "WORK": SignInfo(name="WORK", hand_description="Tap fists together", category="action", difficulty=2),
            "SQUASH_VEGETABLE": SignInfo(name="SQUASH_VEGETABLE", hand_description="Press C-shapes", audio_hint="vegetable", category="food", difficulty=3),
            "SQUASH_SPORT": SignInfo(name="SQUASH_SPORT", hand_description="Hit C-shapes alternately", audio_hint="sport", context_clue="At a sports club", category="sports", difficulty=3),
            "ORANGE_FRUIT": SignInfo(name="ORANGE_FRUIT", hand_description="Squeeze O-shape at chin", audio_hint="fruit", context_clue="In a kitchen", category="food", difficulty=3),
            "ORANGE_COLOR": SignInfo(name="ORANGE_COLOR", hand_description="Squeeze O-shape move forward", audio_hint="color", category="color", difficulty=3),
            "HELLO": SignInfo(name="HELLO", hand_description="Salute from forehead", category="polite", difficulty=1)
        }

    def get_sign(self, name: str) -> Optional[SignInfo]:
        return self.signs.get(name.upper())

class SignInterpreterEnv:
    def __init__(self, max_steps: int = 10, seed: Optional[int] = None):
        self.max_steps = max_steps
        self.dictionary = SignLanguageDictionary()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.stats = {"total_reward": 0.0, "episodes": 0}

    def reset(self, task_id: Optional[int] = None) -> SignObservation:
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Explicit Task ID mapping
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
        else: # HARD - Visual Twin Disambiguation
            pairs = [["BAT", "BASEBALL"], ["SQUASH_VEGETABLE", "SQUASH_SPORT"], ["ORANGE_FRUIT", "ORANGE_COLOR"]]
            return [random.choice(random.choice(pairs))]

    def step(self, action: SignAction) -> Tuple[SignObservation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = -0.01 # Small step penalty
        done = False
        
        if action.action_type == ActionType.SUBMIT_TRANSLATION:
            if action.translation and action.translation.strip().upper() == self.target_translation:
                # Rewards based on difficulty
                reward = {"easy": 1.0, "medium": 1.5, "hard": 2.0}[self.current_difficulty.value]
                done = True
            else:
                reward = -0.2
        
        if self.current_step >= self.max_steps:
            done = True
            
        self.episode_reward += reward
        if done:
            self.stats["total_reward"] += self.episode_reward
            self.stats["episodes"] += 1
            
        return self._get_observation(), reward, done, {}

    def state(self) -> Dict[str, Any]:
        return {
            "difficulty": self.current_difficulty.value,
            "step": self.current_step,
            "target": self.target_translation
        }

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

def get_all_signs():
    return SignLanguageDictionary().signs
