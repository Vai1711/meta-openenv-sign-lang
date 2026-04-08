import asyncio
import os
import json
import sys
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel

# Import environment - ensure this matches your filename
try:
    from env import SignInterpreterEnv, SignAction, ActionType
except ImportError:
    print("Error: Could not import SignInterpreterEnv from env.py", flush=True)
    sys.exit(1)

# MANDATORY ENV VARS - exact names required by grader
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

class SignLanguageAgent:
    """Agent that uses OpenAI to decide actions in Sign Language environment"""
    
    def __init__(self, client: OpenAI, model: str = MODEL_NAME):
        self.client = client
        self.model = model
        self.sign_dictionary = self._load_sign_dictionary()
    
    def _load_sign_dictionary(self) -> Dict[str, str]:
        """Load sign dictionary for context"""
        try:
            from env import get_all_signs
            signs = get_all_signs()
            return {name: f"{info.hand_description} (Category: {info.category})" 
                    for name, info in signs.items()}
        except ImportError:
            return {}
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent"""
        dictionary_info = "\n".join([f"- {name}: {desc[:100]}..." 
                                   for name, desc in list(self.sign_dictionary.items())[:10]])
        
        return f"""You are a Sign Language Interpreter agent. Your task is to translate American Sign Language (ASL) observations into English text.

Available actions:
1. query_dict: Ask about a specific sign to get more information
2. query_context: Ask about context clues for a specific sign
3. submit_translation: Submit your final translation answer

Sign Dictionary (sample):
{dictionary_info}

Instructions:
1. Analyze the hand description, facial expression, and any audio hints
2. For sequences, pay attention to previous signs
3. Use query_dict if you're uncertain about a sign
4. Use query_context for hard difficulty to get context clues
5. Submit your translation when confident
6. For hard difficulty, use audio hints and context clues to disambiguate similar signs

Respond with a JSON object containing:
- action_type: "query_dict", "query_context", or "submit_translation"
- query_sign: (if querying) the sign name to query
- query_context: (if querying context) the sign name to query context for
- translation: (if submitting) your translation answer

Be concise and accurate."""
    
    def _create_observation_prompt(self, obs: Dict[str, Any]) -> str:
        """Create prompt from current observation"""
        prompt = f"""Current Observation:
- Hand Description: {obs.get('hand_description', 'N/A')}
- Facial Expression: {obs.get('facial_expression', 'N/A')}
- Audio Hint: {obs.get('audio_hint', 'N/A')}
- Context Clue: {obs.get('context_clue', 'N/A')}
- Sequence Length: {obs.get('sequence_length', 1)}
- Current Position: {obs.get('current_position', 0)}
- Difficulty: {obs.get('difficulty', 'unknown')}
- Previous Signs: {', '.join(obs.get('previous_signs', []))}

What action should I take?"""
        
        return prompt
    
    def decide_action(self, observation: Dict[str, Any]) -> SignAction:
        """Use OpenAI to decide the next action"""
        
        # Create conversation
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": self._create_observation_prompt(observation)}
        ]
        
        try:
            # Get AI response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                action_data = json.loads(content)
                
                # Create SignAction
                action_type_str = action_data.get("action_type", "submit_translation")
                action_type = ActionType(action_type_str)
                
                return SignAction(
                    action_type=action_type,
                    query_sign=action_data.get("query_sign"),
                    query_context=action_data.get("query_context"),
                    translation=action_data.get("translation")
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback action
                return SignAction(
                    action_type=ActionType.SUBMIT_TRANSLATION,
                    translation="ERROR"
                )
                
        except Exception as e:
            # Fallback action
            return SignAction(
                action_type=ActionType.SUBMIT_TRANSLATION,
                translation="ERROR"
            )

async def main():
    """Main async function for OpenEnv evaluation"""
    
    # Initialize client
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set", flush=True)
        return
    
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Initialize environment
    try:
        env = SignInterpreterEnv(max_steps=10, seed=42)
    except Exception as e:
        print(f"Error initializing environment: {e}", flush=True)
        return
    
    # Initialize agent
    agent = SignLanguageAgent(client, model=MODEL_NAME)
    
    # Logging Helpers strictly following Meta format
    def log_start(task, env_name, model):
        print(f"[START] task={task} env={env_name} model={model}", flush=True)

    def log_step(step, action, reward, done, error):
        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

    def log_end(success, steps, score, rewards):
        r_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r_str}", flush=True)

    # State tracking
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    task_name = os.getenv("TASK_NAME", "sign-translation-hard")

    log_start(task=task_name, env_name="sign-lang-v1", model=MODEL_NAME)

    try:
        # 1. Reset
        obs = env.reset()
        
        for step in range(1, 11):  # MAX_STEPS
            # 2. Get AI Action
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs
            action = agent.decide_action(obs_dict)
            
            # Convert action to string for logging
            if action.action_type == ActionType.SUBMIT_TRANSLATION:
                action_str = f"submit_translation('{action.translation}')"
            elif action.action_type == ActionType.QUERY_DICT:
                action_str = f"query_dict('{action.query_sign}')"
            elif action.action_type == ActionType.QUERY_CONTEXT:
                action_str = f"query_context('{action.query_context}')"
            else:
                action_str = str(action.action_type)
            
            # 3. Step
            new_obs, reward, done, info = env.step(action)
            
            # 4. Mandatory Step Log
            log_step(step, action_str, reward, done, None)
            
            rewards.append(reward)
            steps_taken = step
            if done:
                break
            
            obs = new_obs

        # Calculate final normalized score [0, 1]
        # Normalize based on max possible reward (2.0 for hard difficulty)
        max_possible_reward = 2.0
        score = min(max(sum(rewards) / max_possible_reward, 0.0), 1.0)
        success = score > 0.5

    except Exception as e:
        log_step(steps_taken + 1, "error", 0.0, True, str(e))
        
    finally:
        try:
            await env.close()
        except:
            pass  # Some environments might not have close()
        
        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())
