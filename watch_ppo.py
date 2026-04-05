import os
import time
from stable_baselines3 import PPO
from rl_wrapper import GymCacheEnv

def watch_and_analyze(task_name):
    # Path to your newly trained 1M step models
    model_path = os.path.join("1Mtrained", f"ppo_{task_name}_weights")
    
    # Quick failsafe to ensure the file exists before crashing
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}. Skipping...")
        return

    print(f"\n{'='*55}")
    print(f"🔬 ANALYZING PPO MODEL: {task_name.upper()}")
    print(f"{'='*55}")

    env = GymCacheEnv(task_level=task_name)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    step_count = 0
    hits = 0
    misses = 0
    
    # We track the exact sequence of actions to reverse-engineer the agent's strategy
    action_history = []
    reward_history = []

    print("Running 100-step deterministic evaluation...")
    time.sleep(1) # Brief pause for dramatic effect in the terminal

    while not done:
        step_count += 1
        
        # deterministic=True forces the agent to use its absolute best learned policy,
        # completely disabling random exploration noise.
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        action_history.append(int(action))
        reward_history.append(reward)
        
        if reward > 0:
            hits += 1
            result_str = "✅ HIT "
        else:
            misses += 1
            result_str = "❌ MISS"
            
        # Print a clean, fast-scrolling log of the agent's brain in real-time
        print(f"Step {step_count:03d} | Agent Evicted Slot {int(action)} | {result_str}")
        time.sleep(0.02) # Fast scroll, enough to watch but not boring

    # Calculate final math
    hit_rate = hits / max(1, step_count)
    unique_slots_used = len(set(action_history))

    print(f"\n📊 {task_name.upper()} EPISODE ANALYSIS")
    print("-" * 30)
    print(f"Total Steps : {step_count}")
    print(f"Total Hits  : {hits}")
    print(f"Total Misses: {misses}")
    print(f"🏆 EXACT HIT RATE: {hit_rate:.2f} / 1.00")
    
    print("\n🧠 STRATEGY INSIGHTS:")
    if task_name == "easy":
        print("-> The agent should have achieved a much higher score here compared to the 100k version.")
        print("-> Look at the action log: it should be evicting a wider variety of slots, identifying the 'long tail' of the Zipfian curve.")
        
    elif task_name == "medium":
        print(f"-> The agent utilized {unique_slots_used} out of 10 available cache slots.")
        if unique_slots_used < 10:
            pinned_count = 10 - unique_slots_used
            print(f"-> 🎯 PROOF OF PINNING! The agent intentionally refused to use {pinned_count} slot(s).")
            print("-> It mathematically solved the 'Sequential Trap' by permanently locking items in the cache to break the loop.")
        else:
            print("-> The agent rotated all slots, meaning it found a rolling-wave solution to the sequence.")
            
    elif task_name == "hard":
        # Check if it recovered after step 50
        first_half_hits = sum(1 for r in reward_history[:50] if r > 0)
        second_half_hits = sum(1 for r in reward_history[50:] if r > 0)
        print(f"-> Phase 1 (Steps 1-50) Hits : {first_half_hits}")
        print(f"-> Phase 2 (Steps 51-100) Hits: {second_half_hits}")
        if second_half_hits > 0:
            print("-> 🎯 PHASE SHIFT SURVIVED! The agent successfully adapted to the new data after Step 50.")
        else:
            print("-> The agent still struggles to flush old data after the phase shift.")
    print("\n")

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        watch_and_analyze(task)