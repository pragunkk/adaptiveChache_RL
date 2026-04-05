import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from rl_wrapper import GymCacheEnv

def train_and_benchmark(task_name, steps=100000):
    print(f"\n{'='*50}")
    print(f"🧠 Training PPO Agent on: {task_name.upper()}")
    print(f"{'='*50}")
    
    # 1. Initialize environment
    env = GymCacheEnv(task_level=task_name)
    
    # 2. Build the Neural Network (Multi-Layer Perceptron)
    # 64x64 is a lightweight, highly efficient architecture for 1D arrays
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=dict(net_arch=[64, 64]))
    
    # 3. Train the model
    print(f"Training for {steps} steps... (Please wait 1-2 minutes)")
    model.learn(total_timesteps=steps)
    print("Training Complete!")
    
    # 4. Evaluate the trained agent
    print("Evaluating over 10 deterministic episodes...")
    eval_env = GymCacheEnv(task_level=task_name)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    # 5. Calculate the true Hit Rate
    # Math: Reward = Hits - Misses. Total Steps = Hits + Misses.
    # Therefore: Hits = (Reward + Total Steps) / 2
    # OpenEnv standard episodes are 100 steps long.
    total_steps = 100 
    estimated_hits = (mean_reward + total_steps) / 2
    hit_rate = max(0.0, estimated_hits / total_steps)
    
    print(f"Average Cumulative Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"🏆 FINAL BENCHMARK HIT RATE: {hit_rate:.2f} / 1.00")
    
    # 6. Save the trained weights!
    save_path = f"ppo_{task_name}_weights"
    model.save(save_path)
    print(f"💾 Model saved to {save_path}.zip")
    
    return hit_rate

if __name__ == "__main__":
    easy_score = train_and_benchmark("easy")
    medium_score = train_and_benchmark("medium")
    hard_score = train_and_benchmark("hard")
    
    print("\n" + "="*50)
    print("📊 FINAL PPO BENCHMARKS")
    print("="*50)
    print(f"Easy (Zipfian)   : {easy_score:.2f}")
    print(f"Medium (Sequential): {medium_score:.2f}")
    print(f"Hard (Shifting)  : {hard_score:.2f}")