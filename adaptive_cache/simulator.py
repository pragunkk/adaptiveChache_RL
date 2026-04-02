import numpy as np

class CacheSimulator:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # -1 represents an empty cache slot
        self.cache = np.full(capacity, -1, dtype=np.int32)
        self.last_access_time = np.zeros(capacity, dtype=np.int32)
        self.current_time = 0

    def request_item(self, item_id: int) -> bool:
        """Returns True if hit, False if miss. Does not evict."""
        self.current_time += 1
        
        hit_indices = np.where(self.cache == item_id)[0]
        if len(hit_indices) > 0:
            idx = hit_indices[0]
            self.last_access_time[idx] = self.current_time
            return True
            
        return False

    def evict_and_insert(self, slot_index: int, item_id: int):
        """Places the new item in the specified cache slot."""
        if 0 <= slot_index < self.capacity:
            self.cache[slot_index] = item_id
            self.last_access_time[slot_index] = self.current_time