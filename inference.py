import os
import json
from dotenv import load_dotenv  # <-- ADDED
from openai import OpenAI
from adaptive_cache.env import AdaptiveCacheEnv, Action

# Load variables from .env file into the environment
load_dotenv()  # <-- ADDED

def run_baseline(task_level: str):
    print(f"\n--- Running Baseline for Task: {task_level.upper()} ---")
    
    # 1. Initialize the official OpenAI client dynamically
    # <-- CHANGED: Now pulls from generic LLM_ variables
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model_name = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini") 
    
    if not api_key:
        print("ERROR: LLM_API_KEY environment variable not set. Check your .env file.")
        return

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    env = AdaptiveCacheEnv(task_level=task_level)
    obs = env.reset()
    done = False
    total_reward = 0.0
    
    system_prompt = """
    You are an intelligent Cache Manager. 
    You must decide which cache slot index (0 to 9) to evict.
    Respond ONLY with a JSON object matching this schema: {"evict_index": integer}
    """

    while not done:
        try:
            # 2. Call the dynamically configured model
            # <-- CHANGED: Now uses the model_name variable
            response = client.chat.completions.create(
                model=model_name, 
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current State: {obs.model_dump_json()}"}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            action_dict = json.loads(content)
            action = Action(**action_dict)
            
        except Exception as e:
            # Failsafe so the script doesn't crash on a bad JSON format
            print(f"LLM Parsing failed ({e}). Defaulting to slot 0.")
            action = Action(evict_index=0)
            
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
    print(f"Episode Finished.")
    print(f"Total Reward: {total_reward}")
    print(f"Final Grader Score (Hit Rate): {info.get('score', 0.0):.2f} / 1.00")

if __name__ == "__main__":
    run_baseline("easy")
    run_baseline("medium")
    run_baseline("hard")