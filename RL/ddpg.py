import os
import torch
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from f1tenth_gym_ros.simple_rl import F110Gym

log_dir = "logs/DDPG_F1Tenth"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()

n_actions = env.action_space.shape[-1]  
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

new_logger = configure(log_dir, ["stdout", "tensorboard"])

model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log=log_dir,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
model.set_logger(new_logger)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=os.path.join(log_dir, "checkpoints"),
    name_prefix="ddpg_f1tenth_checkpoint",
    save_replay_buffer=True,
)

callback = CallbackList([checkpoint_callback])

model.learn(
    total_timesteps=5_000_000,
    callback=callback,
    progress_bar=True
)

model.save(os.path.join(log_dir, "ddpg_f1tenth_final"))