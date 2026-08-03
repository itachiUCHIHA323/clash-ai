"""
Random Baseline Agent
---------------------
Runs a random deployment strategy to test the Gymnasium environment and ADB bridge.
"""

import time
from src.env.clash_env import ClashOfClansEnv


def run_random_agent(episodes: int = 2, max_steps: int = 20):
    env = ClashOfClansEnv(max_steps=max_steps)
    print("[INFO] Starting random baseline agent test...")

    for ep in range(episodes):
        obs, info = env.reset()
        print(f"\n--- Episode {ep + 1} Started ---")

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Step {step + 1}: Action={action} -> Reward={reward:.2f} | "
                f"Destruction={info['destruction_percentage']:.1f}% | Stars={info['stars']}"
            )
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}.")
                break
        time.sleep(1)

    env.close()
    print("[INFO] Random agent test complete.")


if __name__ == "__main__":
    run_random_agent()
