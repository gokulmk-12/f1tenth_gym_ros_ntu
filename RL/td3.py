import os
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

from f1tenth_gym_ros.simple_rl import F110Gym
import numpy as np

exp_no = 1
exp_name = "obs_logs"
log_dir = f"logs/{exp_name}/TD3_F1Tenth_Exp{exp_no}"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()
env.reset()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

weights_path = os.path.join(log_dir, "td3_f1tenth_final.zip")
device = "cuda" if torch.cuda.is_available() else "cpu"

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

if os.path.exists(weights_path):
    print(f"Loading model from {weights_path}")
    model = TD3.load(weights_path, env=env, device=device)
else:
    print("No saved model found, initializing new TD3 model")
    model = TD3(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        device=device,
        learning_rate=5e-4,
        buffer_size=1_000_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        action_noise=action_noise
    )

model.set_logger(new_logger)

model.learn(
    total_timesteps=1_000_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "td3_f1tenth_final"))
