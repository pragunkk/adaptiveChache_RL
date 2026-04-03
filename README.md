---
title: Adaptive Cache Manager
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - agents
---

# 🧠 Adaptive Cache Manager (OpenEnv)

An OpenEnv-compliant reinforcement learning and agentic AI environment that simulates a high-performance operating system memory manager. 

Instead of relying on static, heuristic-based algorithms like LRU (Least Recently Used) or LFU (Least Frequently Used), this environment challenges frontier AI agents to dynamically learn and execute optimal cache eviction policies against complex, shifting workloads.

## 🌍 Real-World Utility & Motivation
Every modern operating system, database management system (DBMS), and CDN relies heavily on cache efficiency. A 1% increase in cache hit rates can save massive amounts of compute, bandwidth, and energy. 

However, standard algorithms fail when traffic patterns change abruptly or fall into sequential loops. This environment isolates that specific, high-value DevOps/DBA problem. It moves away from "toy" text-parsing tasks and provides a pure, mathematically grounded testbed for reasoning models and RL agents to prove their algorithmic optimization capabilities.

---

## 🛠 Environment Design: Spaces & Rewards

The environment strictly implements the OpenEnv API via typed Pydantic models and exposes standard `POST /reset` and `POST /step` web endpoints via FastAPI.

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
* **`+1.0`** for every Cache Hit.
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


## 📊 Baseline Comparisons

To demonstrate the necessity of intelligent eviction policies, this environment provides benchmark scores comparing traditional operating system algorithms against a zero-shot LLM baseline (Llama-3 8B). The table below displays the final **Hit Rate (0.0 to 1.0)**.

| Task (Workload) | Random Eviction | LRU | LFU | LLM Agent (Zero-Shot) |
| :--- | :--- | :--- | :--- | :--- |
| **Easy (Zipfian)** | 0.64 | 0.18 | 0.44 | **0.67** |
| **Medium (Sequential)** | 0.35 | 0.00 | 0.08 | **0.16** |
| **Hard (Shifting)** | **0.35** | 0.04 | 0.13 | 0.12 |

**Key Insights for Researchers:**
* **The Sequential Trap:** As proven by the Medium task, standard LRU algorithms achieve a mathematical **0.00 hit rate** when faced with sequence loops larger than the cache size. The LLM demonstrates foundational reasoning to break this loop, outperforming both LRU and LFU.
* **The Shifting Challenge:** The Hard task proves that static frequency counters (LFU) and smaller zero-shot LLMs both struggle to adapt to sudden data shifts. This sets a clear, rigorous benchmark for future Reinforcement Learning agents to conquer.

---

## 🚀 Setup & Execution

### 1. Local Setup (Modern `uv` package manager)
This project uses modern Python packaging via `pyproject.toml` and `uv.lock`.

```bash
# Install the ultra-fast uv package manager
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
uv sync
```

```bash
#create .env file in root directory
LLM_API_KEY="model api key"
LLM_BASE_URL="model api url"
LLM_MODEL_NAME="model name"
```

### 2. Running the Inference Agent
The inference.py script evaluates the environment using a zero-shot LLM baseline via the official OpenAI Python SDK.

(Note: To ensure tests can be run repeatedly without cost during development, the script reads from the strict OPENAI_API_KEY variable as per OpenEnv specs, but the base URL can be pointed to Groq's free models).

```bash
# Export your API key
export GROQ_API_KEY="your-api-key-here"

# Run the baseline evaluation across all 3 tasks
python inference.py
```

### 3. Docker & Hugging Face Deployment
This environment is fully containerized, web-server enabled (FastAPI/Uvicorn), and designed for multi-mode deployment as a Hugging Face Space.

```bash
# Build the image locally
docker build -t adaptive-cache-env .

# Run the container locally (boots the FastAPI server on port 7860)
docker run -p 7860:7860 adaptive-cache-env
```

## 📂 Project Structure

```bash
adaptive-cache-env/
├── Dockerfile             # Container configuration pointing to server.app
├── pyproject.toml         # Modern build system & OpenEnv core dependencies
├── uv.lock                # Strict dependency lockfile
├── openenv.yaml           # OpenEnv task and metadata specifications
├── inference.py           # Baseline LLM inference script
├── test_env.py            # Deterministic grader bounds validation
├── README.md              # Project documentation
├── server/
│   └── app.py             # FastAPI web server and OpenEnv POST endpoints
└── adaptive_cache/
    ├── __init__.py
    ├── simulator.py       # Core OS-level array and memory simulation
    ├── workloads.py       # Deterministic task generators (Zipfian, Sequential, etc.)
    └── env.py             # OpenEnv wrapper and Pydantic models

```