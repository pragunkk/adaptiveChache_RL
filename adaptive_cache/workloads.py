import numpy as np

def generate_easy_task(length=100, vocab_size=50):
    """Zipfian (power-law) distribution. Standard web traffic."""
    np.random.seed(42)
    workload = np.random.zipf(1.5, length)
    return np.clip(workload, 1, vocab_size).tolist()

def generate_medium_task(length=100, cache_size=10):
    """Sequential scan loop. Defeats standard LRU."""
    sequence = list(range(1, cache_size + 3)) 
    return (sequence * (length // len(sequence) + 1))[:length]

def generate_hard_task(length=100):
    """Shifting working sets. Requires rapid adaptation."""
    np.random.seed(42)
    first_half = np.random.randint(1, 20, length // 2).tolist()
    second_half = np.random.randint(80, 100, length - (length // 2)).tolist()
    return first_half + second_half