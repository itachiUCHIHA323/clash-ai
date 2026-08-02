"""
PPO Training Script
-------------------
Trains a Proximal Policy Optimization (PPO) reinforcement learning agent
to play Clash of Clans using Stable-Baselines3.
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.env.clash_env import ClashOfClansEnv


def train(total_timesteps: int = 100000, device_serial: str = None, save_dir: str = "./models"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Initializing ClashOfClansEnv (Device: {device_serial or 'Default ADB'})...")
    env = ClashOfClansEnv(device_serial=device_serial)

    print("[INFO] Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./tensorboard_logs/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path=save_dir,
        name_prefix="ppo_clash_ai",
    )

    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    print("[INFO] Training complete!")

    final_model_path = os.path.join(save_dir, "ppo_clash_ai_final.zip")
    model.save(final_model_path)
    print(f"[INFO] Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on Clash of Clans")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="Number of timesteps to train")
    parser.add_argument("--device-serial", type=str, default=None, help="ADB Device/Emulator serial number")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save checkpoint models")
    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        device_serial=args.device_serial,
        save_dir=args.save_dir,
    )
