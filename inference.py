import asyncio
import os
import json
import sys
from typing import Dict, Any
from openai import OpenAI
from env import SignInterpreterEnv, SignAction, ActionType

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

class SignLanguageAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = MODEL_NAME

    def decide_action(self, obs: Dict[str, Any]) -> SignAction:
        prompt = f"ASL Observation: {obs}. Disambiguate if Hard. Reply JSON: {{'action_type': 'submit_translation', 'translation': 'WORD'}}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        return SignAction(**data)

async def main():
    if not HF_TOKEN: return
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SignInterpreterEnv(max_steps=10)
    agent = SignLanguageAgent(client)

    # Resolve task from environment variable or default to Task 0
    task_name = os.getenv("TASK_NAME", "sign-translation-easy")
    task_id_map = {"sign-translation-easy": 0, "sign-translation-medium": 1, "sign-translation-hard": 2}
    task_id = task_id_map.get(task_name, 0)

    print(f"[START] task={task_name}")
    obs = env.reset(task_id=task_id)
    rewards = []

    for step in range(1, 11):
        action = agent.decide_action(obs.model_dump())
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        print(f"[STEP] step={step} reward={reward:.2f}")
        if done: break

    # FIX: Strict Range Jittering (0.01 to 0.99)
    total_r = sum(rewards)
    max_r = {0: 1.0, 1: 1.5, 2: 2.0}[task_id]
    
    # Normalize and clamp strictly between 0 and 1
    normalized_score = max(0.0, min(1.0, total_r / max_r))
    final_score = 0.01 + (normalized_score * 0.98) # Scales 0->1 into 0.01->0.99
    
    print(f"[END] score={final_score:.3f}")

if __name__ == "__main__":
    async def run():
        try: await main()
        except Exception as e: print(f"Error: {e}")
    asyncio.run(run())
