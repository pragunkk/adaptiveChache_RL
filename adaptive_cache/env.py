from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from .simulator import CacheSimulator
from .workloads import generate_easy_task, generate_medium_task, generate_hard_task

class Observation(BaseModel):
    incoming_request: int = Field(description="The ID of the data item being requested.")
    incoming_cost: float = Field(description="The latency cost in ms to fetch this item if missed.")
    cache_state: List[int] = Field(description="Current items in the cache. -1 means empty.")
    idle_times: List[int] = Field(description="Time steps since each cache slot was last accessed.")

class Action(BaseModel):
    evict_index: int = Field(description="The index (0 to capacity-1) of the cache slot to evict.")

class AdaptiveCacheEnv:
    def __init__(self, task_level: str = "easy", capacity: int = 10):
        self.capacity = capacity
        self.task_level = task_level
        self.sim = CacheSimulator(capacity)
        
        if task_level == "easy":
            self.workload = generate_easy_task()
        elif task_level == "medium":
            self.workload = generate_medium_task(cache_size=capacity)
        else:
            self.workload = generate_hard_task()
            
        self.step_count = 0
        self.hits = 0

    def reset(self) -> Observation:
        self.sim = CacheSimulator(self.capacity)
        self.step_count = 0
        self.hits = 0
        return self.state()

    def state(self) -> Observation:
        # Safe check for the terminal state to prevent IndexError
        if self.step_count >= len(self.workload):
            current_item = -1  # Simulation is over, no more incoming requests
            current_cost = 0.0
        else:
            current_item, current_cost = self.workload[self.step_count]
            
        idle_times = [(self.sim.current_time - t) if t > 0 else 0 for t in self.sim.last_access_time]
        return Observation(
            incoming_request=current_item,
            incoming_cost=current_cost,
            cache_state=self.sim.cache.tolist(),
            idle_times=idle_times
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        # 1. Apply Action (Evict and Insert)
        current_item, current_cost = self.workload[self.step_count]
        self.sim.evict_and_insert(action.evict_index, current_item)
        
        # 2. Advance time strictly by 1 step
        self.step_count += 1
        
        # 3. Check Episode Boundary
        done = self.step_count >= len(self.workload)
        reward = 0.0
        
        if done:
            final_score = self.hits / max(1, len(self.workload))
            return self.state(), reward, True, {"score": final_score}

        # 4. Evaluate the *next* state strictly without fast-forwarding
        next_item, next_cost = self.workload[self.step_count]
        is_hit = self.sim.request_item(next_item)
        
        # --- THE HACKATHON FLEX: Latency-Based Reward Math ---
        # Normalize the cost so rewards stay between -1.0 and +1.0
        normalized_cost = next_cost / 100.0
        
        if is_hit:
            reward = normalized_cost 
            self.hits += 1
        else:
            reward = -normalized_cost 
            
        current_score = self.hits / max(1, self.step_count)
        info = {"score": current_score, "hits": self.hits, "steps": self.step_count, "latency_cost": next_cost}
        
        return self.state(), reward, done, info