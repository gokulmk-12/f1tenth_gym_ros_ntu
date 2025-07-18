import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from f1tenth_gym_ros.simple_rl import F110Gym

log_dir = "logs/PPO_F1Tenth"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=log_dir,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=os.path.join(log_dir, "checkpoints"),
    name_prefix="ppo_f1tenth_checkpoint",
)

callback = CallbackList([checkpoint_callback])

model.learn(
    total_timesteps=2_000_000,
    callback=callback,
    progress_bar=True
)

model.save(os.path.join(log_dir, "ppo_f1tenth_final"))