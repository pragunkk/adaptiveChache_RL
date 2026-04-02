# 🧠 Adaptive Cache Manager (OpenEnv)

An OpenEnv-compliant reinforcement learning and agentic AI environment that simulates a high-performance operating system memory manager. 

Instead of relying on static, heuristic-based algorithms like LRU (Least Recently Used) or LFU (Least Frequently Used), this environment challenges frontier AI agents to dynamically learn and execute optimal cache eviction policies against complex, shifting workloads.

## 🌍 Real-World Utility & Motivation
Every modern operating system, database management system (DBMS), and CDN relies heavily on cache efficiency. A 1% increase in cache hit rates can save massive amounts of compute, bandwidth, and energy. 

However, standard algorithms fail when traffic patterns change abruptly or fall into sequential loops. This environment isolates that specific, high-value DevOps/DBA problem. It moves away from "toy" text-parsing tasks and provides a pure, mathematically grounded testbed for reasoning models and RL agents to prove their algorithmic optimization capabilities.

---

## 🛠 Environment Design: Spaces & Rewards

The environment strictly implements the OpenEnv API via typed Pydantic models.

### Observation Space
The agent receives a lightweight, numerical snapshot of the memory system at the exact moment a cache miss occurs.
* `incoming_request` (int): The ID of the data item currently requested by the system.
* `cache_state` (List[int]): The current items residing in the cache slots (-1 indicates an empty slot).
* `idle_times` (List[int]): The number of timesteps since each specific cache slot was last accessed.

### Action Space
The agent must decide which slot to free up.
* `evict_index` (int): A discrete integer (0 to capacity-1) representing the index of the cache slot to overwrite.

### Reward Function
The environment provides a dense, step-by-step reward signal directly correlated to system performance:
* **`+1.0`** for every Cache Hit (including consecutive hits safely fast-forwarded without agent intervention).
* **`-1.0`** for a Cache Miss (forcing the agent to step in and evict).

---

## 🏆 Tasks & Difficulty Progression

The environment features three programmatic workloads (tasks) designed to challenge agents with distinctly different access patterns. The **Grader** for all tasks deterministically calculates the final **Hit Rate (0.0 to 1.0)**.

1. **`cache-zipfian-easy` (Easy)**
   * **Workload:** A Zipfian (power-law) distribution simulating standard web traffic. A few items are requested constantly; a long tail is requested rarely.
   * **Goal:** Outperform random eviction by pinning the most frequently requested items.

2. **`cache-sequential-medium` (Medium)**
   * **Workload:** A looping sequential scan (e.g., requesting items 1 through 12 in a loop for a cache of size 10). 
   * **Goal:** Standard LRU algorithms achieve a **0% hit rate** here. The agent must break static logic and learn to pin a subset of the sequence to guarantee hits.

3. **`cache-shifting-hard` (Hard)**
   * **Workload:** Abruptly shifting working sets. The first half heavily favors one block of data; the second half abruptly shifts entirely to a different block.
   * **Goal:** Requires rapid, aggressive adaptation to flush obsolete items. Often acts as a stumbling block for zero-shot LLMs, requiring true RL or deep reasoning.

---

## 🚀 Setup & Execution

### 1. Local Virtual Environment Setup
Ensure you are using Python 3.10 or higher (Python 3.13 is fully supported).

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Baseline Agent
The baseline script uses Groq's Llama-3 model to evaluate the environment via the official OpenAI Python SDK, satisfying the OpenEnv API client requirement while remaining 100% free and lightning-fast.

```bash
# Export your free Groq API key (get one at console.groq.com)
export GROQ_API_KEY="your-api-key-here"

# Run the baseline evaluation across all 3 tasks
python baseline.py
```

### 3. Docker & Hugging Face Deployment
This environment is fully containerized and designed for deployment as a Hugging Face Space.

```bash
# Build the image
docker build -t adaptive-cache-env .

# Run the container (pass your API key)
docker run -e GROQ_API_KEY="your-api-key-here" adaptive-cache-env
```

## 📂 Project Structure

```bash
adaptive-cache-env/
├── Dockerfile             # Container configuration for HF Spaces
├── requirements.txt       # Project dependencies (NumPy 2.x, Pydantic, OpenAI SDK)
├── openenv.yaml           # OpenEnv task and metadata specifications
├── baseline.py            # Baseline LLM inference script
├── README.md              # Project documentation
└── adaptive_cache/
    ├── __init__.py
    ├── simulator.py       # Core OS-level array and memory simulation
    ├── workloads.py       # Deterministic task generators (Zipfian, Sequential, etc.)
    └── env.py             # OpenEnv wrapper and Pydantic models
```