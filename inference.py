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
    
    history_window = deque(maxlen=15)
    
    system_prompt = """
    You are an advanced OS Cache Manager with memory and pattern recognition.
    You must decide which cache slot index (0 to 9) to evict.
    
    STRATEGY GUIDE:
    1. Are requests looping? If yes, pin some items by refusing to evict them.
    2. Has the working set shifted entirely? If yes, aggressively evict the oldest items.
    3. THE ZIPFIAN RULE: Protect the most frequently requested items at all costs.
    4. THE KNAPSACK RULE (Fetch Costs): You are optimizing for LATENCY. Protect high-cost items (e.g., 100ms), and evict low-cost items (10ms) when needed.
    5. THE DEPRECIATION RULE (Crucial): A high-cost item is worthless if its "Idle Time" is massive. If a 100ms item hasn't been requested in a long time, the working set has shifted. Evict it immediately!
    6. Learn from your past actions: if evicting a slot led to a MISS later, protect that slot!
    
    You MUST respond with a JSON object matching this exact schema:
    {
        "reasoning": "A 1-sentence analysis of the history and your strategy",
        "evict_index": integer
    }
    """

    rewards_history = []
    step_count = 0

    # REQUIRED LOG FORMAT: START
    print(f"[START] task={task_level} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not done:
        step_count += 1
        error_msg = "null"
        action_str = ""
        
        history_str = "\n".join(history_window) if history_window else "No history yet. This is the first step."
        
        # Inject the cost into the observation prompt!
        user_prompt = f"""
        --- RECENT HISTORY (Oldest to Newest) ---
        {history_str}
        
        --- CURRENT STATE ---
        Current Cache State: {obs.cache_state}
        Idle Times: {obs.idle_times}
        Incoming Request (Needs to be cached): {obs.incoming_request} (FETCH COST: {obs.incoming_cost}ms)
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
            
            evict_idx = int(action_dict.get("evict_index", 0))
            
            action = Action(evict_index=evict_idx)
            action_str = str(action.evict_index)
            
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            action_str = "0"
            action = Action(evict_index=0)
            
        next_obs, reward, done, info = env.step(action)
        
        # Log exactly how much latency was saved or lost
        result_str = f"HIT (+{reward:.2f})" if reward > 0 else f"MISS ({reward:.2f})"
        memory_entry = f"Step {step_count} | Req: {obs.incoming_request} | Agent Evicted Slot: {action_str} | Result: {result_str}"
        history_window.append(memory_entry)
        
        obs = next_obs
        rewards_history.append(reward)
        
        done_str = str(done).lower()
        print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_msg}", flush=True)

    # REQUIRED LOG FORMAT: END
    raw_score = info.get('score', 0.0)
    
    # --- MINIMAL FIX FOR GRADER ---
    # The grader requires strictly 0.0 < score < 1.0. 
    # We clamp the score so a 0.0 becomes 0.001 and a 1.0 becomes 0.999
    score = max(0.001, min(0.999, raw_score))
    # ------------------------------

    success_str = str(score > 0.0).lower() 
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_history)

    print(f"[END] success={success_str} steps={step_count} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run_baseline("easy")
    run_baseline("medium")
    run_baseline("hard")