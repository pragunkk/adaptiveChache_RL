from fastapi import FastAPI
from adaptive_cache.env import AdaptiveCacheEnv
import uvicorn

app = FastAPI(title="Adaptive Cache Manager OpenEnv")
env = AdaptiveCacheEnv()

@app.get("/")
def read_root():
    return {
        "status": "Online", 
        "environment": "Adaptive Cache Manager",
        "openenv_compliant": True
    }

@app.get("/reset")
def reset_env():
    obs = env.reset()
    return {"observation": obs.model_dump()}

if __name__ == "__main__":
    # Port 7860 is the mandatory default port for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)