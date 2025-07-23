import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from f1tenth_gym_ros.simple_rl import F110Gym

exp_no = 3
log_dir = f"logs/PPO_F1Tenth_Exp{exp_no}"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

weights_path = os.path.join(log_dir, "ppo_f1tenth_final.zip")
if os.path.exists(weights_path):
    print(f"Loading model from {weights_path}")
    model = PPO.load(weights_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    print("No saved model found, initializing new PPO model")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
model.set_logger(new_logger)

model.learn(
    total_timesteps=2_000_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "ppo_f1tenth_final"))