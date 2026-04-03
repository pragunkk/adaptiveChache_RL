import os
import json
from collections import deque
from dotenv import load_dotenv
from openai import OpenAI
from adaptive_cache.env import AdaptiveCacheEnv, Action

# Load variables from local .env file
load_dotenv()

# STRICT COMPLIANCE: Match the pre-submission checklist exactly
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "adaptive-cache" 

def run_baseline(task_level: str):
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", flush=True)
        return

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )
    
    env = AdaptiveCacheEnv(task_level=task_level)
    obs = env.reset()
    done = False
    
    # ---------------------------------------------------------
    # PHASE 2 UPGRADE: Agentic Memory Trackers
    # ---------------------------------------------------------
    # We keep the last 15 steps of history. 
    # If the sequence loop is 12 items long, 15 gives the LLM 
    # enough vision to realize the pattern is repeating.
    history_window = deque(maxlen=15)
    
    system_prompt = """
    You are an advanced OS Cache Manager with memory and pattern recognition.
    You must decide which cache slot index (0 to 9) to evict.
    
    STRATEGY GUIDE:
    1. Look at the "Recent History". If you see requests looping (e.g., 1, 2, 3... 1, 2, 3), DO NOT use standard LRU. Pin some items by refusing to evict them.
    2. If you see a sudden shift to entirely new request numbers, aggressively evict the oldest items.
    3. Learn from your past actions: if evicting a slot led to a MISS later, protect that slot!
    
    Respond ONLY with a JSON object matching this schema: {"evict_index": integer}
    """

    rewards_history = []
    step_count = 0

    # REQUIRED LOG FORMAT: START
    print(f"[START] task={task_level} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not done:
        step_count += 1
        error_msg = "null"
        action_str = ""
        
        # Format the memory for the LLM
        history_str = "\n".join(history_window) if history_window else "No history yet. This is the first step."
        
        user_prompt = f"""
        --- RECENT HISTORY (Oldest to Newest) ---
        {history_str}
        
        --- CURRENT STATE ---
        Current Cache State: {obs.cache_state}
        Idle Times: {obs.idle_times}
        Incoming Request (Needs to be cached): {obs.incoming_request}
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            action_dict = json.loads(content)
            action = Action(**action_dict)
            action_str = str(action.evict_index)
            
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            action_str = "0"
            action = Action(evict_index=0)
            
        # Step the environment
        next_obs, reward, done, info = env.step(action)
        
        # ---------------------------------------------------------
        # PHASE 2 UPGRADE: Log the outcome into memory
        # ---------------------------------------------------------
        # We record what was requested, what the agent did, and if it worked.
        result_str = "HIT (+1.0)" if reward > 0 else "MISS (-1.0)"
        memory_entry = f"Step {step_count} | Req: {obs.incoming_request} | Agent Evicted Slot: {action_str} | Result: {result_str}"
        history_window.append(memory_entry)
        
        # Update observation for the next loop
        obs = next_obs
        rewards_history.append(reward)
        
        # REQUIRED LOG FORMAT: STEP
        done_str = str(done).lower()
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}", flush=True)

    # REQUIRED LOG FORMAT: END
    score = info.get('score', 0.0)
    success_str = str(score > 0.0).lower() 
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_history)

    print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_baseline("easy")
    run_baseline("medium")
    run_baseline("hard")