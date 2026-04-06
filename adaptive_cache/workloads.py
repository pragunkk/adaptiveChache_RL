import numpy as np

def _assign_costs(requests):
    """
    Helper function to assign a deterministic fetch cost to each item.
    Costs range from 10ms to 100ms. 
    E.g., item 1 -> 20ms, item 9 -> 100ms.
    """
    return [(int(req), float((req % 10) * 10 + 10)) for req in requests]

def generate_easy_task(length=100, vocab_size=50):
    """Zipfian (power-law) distribution. Standard web traffic."""
    np.random.seed(42)
    workload = np.random.zipf(1.5, length)
    requests = np.clip(workload, 1, vocab_size).tolist()
    return _assign_costs(requests)

def generate_medium_task(length=100, cache_size=10):
    """Sequential scan loop. Defeats standard LRU."""
    sequence = list(range(1, cache_size + 3)) 
    requests = (sequence * (length // len(sequence) + 1))[:length]
    return _assign_costs(requests)

def generate_hard_task(length=100):
    """Shifting working sets. Requires rapid adaptation."""
    np.random.seed(42)
    first_half = np.random.randint(1, 20, length // 2).tolist()
    second_half = np.random.randint(80, 100, length - (length // 2)).tolist()
    return _assign_costs(first_half + second_half)