import random
import numpy as np
from adaptive_cache.env import AdaptiveCacheEnv, Action

def run_random_agent(task_name):
    """Evicts a random cache slot."""
    # FIXED: Passed task_name to the correct 'task_level' argument
    env = AdaptiveCacheEnv(task_level=task_name)
    obs = env.reset()
    done = False
    
    while not done:
        capacity = len(obs.cache_state)
        # Pick a random slot index to overwrite
        action = Action(evict_index=random.randint(0, capacity - 1))
        obs, reward, done, info = env.step(action)
        
    return info.get("score", 0.0)

def run_lru_agent(task_name):
    """Evicts the slot with the highest idle time."""
    # FIXED: Passed task_name to the correct 'task_level' argument
    env = AdaptiveCacheEnv(task_level=task_name)
    obs = env.reset()
    done = False
    
    while not done:
        # np.argmax returns the index of the highest value in the array
        # The highest idle_time is our Least Recently Used item
        evict_idx = int(np.argmax(obs.idle_times))
        action = Action(evict_index=evict_idx)
        obs, reward, done, info = env.step(action)
        
    return info.get("score", 0.0)

def run_lfu_agent(task_name):
    """Evicts the slot containing the least frequently requested item."""
    # FIXED: Passed task_name to the correct 'task_level' argument
    env = AdaptiveCacheEnv(task_level=task_name)
    obs = env.reset()
    done = False
    
    # Dictionary to track the global frequency of all requested items
    frequencies = {}
    
    while not done:
        req = obs.incoming_request
        if req != -1:
            # Increment the frequency counter for the incoming request
            frequencies[req] = frequencies.get(req, 0) + 1
            
        cache = obs.cache_state
        best_evict_idx = 0
        min_freq = float('inf')
        
        # Scan the cache to find the item with the lowest frequency
        for i, item in enumerate(cache):
            if item == -1: 
                # If there is an empty slot, always choose it first
                best_evict_idx = i
                break
                
            freq = frequencies.get(item, 0)
            if freq < min_freq:
                min_freq = freq
                best_evict_idx = i
                
        action = Action(evict_index=best_evict_idx)
        obs, reward, done, info = env.step(action)
        
    return info.get("score", 0.0)

if __name__ == "__main__":
    # FIXED: The array now uses the exact strings your if/elif block expects
    tasks = ["easy", "medium", "hard"]
    
    print("==========================================")
    print("🚀 Running Traditional OS Baselines")
    print("==========================================\n")
    
    for task in tasks:
        print(f"Task: {task.upper()}")
        print("-" * 40)
        
        rnd_score = run_random_agent(task)
        print(f"🎲 Random Eviction Hit Rate: {rnd_score:.2f}")
        
        lru_score = run_lru_agent(task)
        print(f"🕒 LRU (Least Recently Used): {lru_score:.2f}")
        
        lfu_score = run_lfu_agent(task)
        print(f"📊 LFU (Least Frequently Used): {lfu_score:.2f}\n")