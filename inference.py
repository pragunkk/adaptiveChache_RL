import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from adaptive_cache.env import AdaptiveCacheEnv, Action

# Load variables from local .env file (for local testing)
load_dotenv()

# 1. STRICT COMPLIANCE: Match the pre-submission checklist exactly
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "adaptive-cache" 

def run_baseline(task_level: str):
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", flush=True)
        return

    # Pass the HF_TOKEN to the api_key parameter of the OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )
    
    env = AdaptiveCacheEnv(task_level=task_level)
    obs = env.reset()
    done = False
    
    system_prompt = """
    You are an intelligent Cache Manager. 
    You must decide which cache slot index (0 to 9) to evict.
    Respond ONLY with a JSON object matching this schema: {"evict_index": integer}
    """

    # Trackers required for the grader's END log
    rewards_history = []
    step_count = 0

    # 2. REQUIRED LOG FORMAT: START
    print(f"[START] task={task_level} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not done:
        step_count += 1
        error_msg = "null"
        action_str = ""
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
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
            action_str = str(action.evict_index)
            
        except Exception as e:
            # Format error to be strictly on a single line for the grader
            error_msg = str(e).replace('\n', ' ')
            action_str = "0"
            action = Action(evict_index=0)
            
        obs, reward, done, info = env.step(action)
        rewards_history.append(reward)
        
        # 3. REQUIRED LOG FORMAT: STEP
        done_str = str(done).lower()
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}", flush=True)

    # 4. REQUIRED LOG FORMAT: END
    score = info.get('score', 0.0)
    success_str = str(score > 0.0).lower() 
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_history)

    print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_baseline("easy")
    run_baseline("medium")
    run_baseline("hard")