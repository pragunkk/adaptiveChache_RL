from stable_baselines3 import PPO
from rl_wrapper import GymCacheEnv
import time

def watch_agent(task_name):
    print(f"\n--- Loading Trained PPO Agent for {task_name.upper()} ---")
    
    env = GymCacheEnv(task_level=task_name)
    model = PPO.load(f"ppo_{task_name}_weights")
    
    obs, _ = env.reset()
    done = False
    step_count = 0
    
    while not done:
        step_count += 1
        # Predict the best action deterministically (no random guessing)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        result = "✅ HIT" if reward > 0 else "❌ MISS"
        print(f"Step {step_count:03d} | Agent Evicted Slot: {int(action)} | {result}")
        time.sleep(0.05) # Slow it down so you can watch it

if __name__ == "__main__":
    # Watch it crush the medium task!
    watch_agent("medium")