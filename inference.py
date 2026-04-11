import asyncio
import os
import json
import sys
from typing import Dict, Any
from openai import OpenAI
from env import SignInterpreterEnv, SignAction, ActionType

# MANDATORY ENV VARS
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

class SignLanguageAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = MODEL_NAME

    def decide_action(self, obs: Dict[str, Any]) -> SignAction:
        # Improved prompt to ensure the LLM knows it's an ASL interpreter
        prompt = (
            f"You are an ASL Interpreter. Observation: {obs}. "
            "If difficulty is 'hard', use audio_hint and context_clue to disambiguate. "
            "If difficulty is 'medium', remember this is a sequence. "
            "Reply strictly in JSON: {'action_type': 'submit_translation', 'translation': 'WORD'}"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        return SignAction(**data)

async def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set")
        return
        
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SignInterpreterEnv(max_steps=10)
    agent = SignLanguageAgent(client)

    # Task mapping for the grader
    task_name = os.getenv("TASK_NAME", "sign-translation-easy")
    task_id_map = {"sign-translation-easy": 0, "sign-translation-medium": 1, "sign-translation-hard": 2}
    task_id = task_id_map.get(task_name, 0)

    # [START] log is mandatory
    print(f"[START] task={task_name} env=sign-lang-v1 model={MODEL_NAME}", flush=True)
    
    obs = env.reset(task_id=task_id)
    rewards = []
    steps_taken = 0

    for step in range(1, 11):
        steps_taken = step
        # Convert observation to dict for the agent
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs
        
        action = agent.decide_action(obs_dict)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        
        # [STEP] log is mandatory
        print(f"[STEP] step={step} action={action.action_type} reward={reward:.2f} done={str(done).lower()}", flush=True)
        
        if done:
            break

    # Final Score Calculation
    total_r = sum(rewards)
    # Match these max rewards to the logic in env.py
    max_r = {0: 1.0, 1: 1.5, 2: 2.0}[task_id]
    
    # Normalize and jitter to strictly (0.01, 0.99)
    normalized_score = max(0.0, min(1.0, total_r / max_r))
    final_score = 0.01 + (normalized_score * 0.98)
    success = final_score > 0.5

    # [END] log is mandatory and must include these 4 fields
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    async def run():
        try:
            await main()
        except Exception as e:
            print(f"[STEP] step=1 action=error reward=0.00 done=true error={str(e)}", flush=True)
    asyncio.run(run())
