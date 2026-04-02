from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from .simulator import CacheSimulator
from .workloads import generate_easy_task, generate_medium_task, generate_hard_task

class Observation(BaseModel):
    incoming_request: int = Field(description="The ID of the data item being requested.")
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
        current_item = self.workload[self.step_count]
        idle_times = [(self.sim.current_time - t) if t > 0 else 0 for t in self.sim.last_access_time]
        return Observation(
            incoming_request=current_item,
            cache_state=self.sim.cache.tolist(),
            idle_times=idle_times
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        current_item = self.workload[self.step_count]
        
        self.sim.evict_and_insert(action.evict_index, current_item)
        self.step_count += 1
        
        done = self.step_count >= len(self.workload) - 1
        reward = 0.0
        
        if done:
            final_score = self.hits / max(1, len(self.workload))
            return self.state(), reward, True, {"score": final_score}

        next_item = self.workload[self.step_count]
        is_hit = self.sim.request_item(next_item)
        
        if is_hit:
            reward = 1.0 
            self.hits += 1
            while is_hit and not done:
                self.step_count += 1
                done = self.step_count >= len(self.workload) - 1
                if not done:
                    next_item = self.workload[self.step_count]
                    is_hit = self.sim.request_item(next_item)
                    if is_hit:
                        self.hits += 1
                        reward += 1.0 
        else:
            reward = -1.0 
            
        current_score = self.hits / max(1, self.step_count)
        info = {"score": current_score, "hits": self.hits, "steps": self.step_count}
        
        return self.state(), reward, done, info