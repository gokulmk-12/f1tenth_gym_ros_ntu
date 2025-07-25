import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from f1tenth_gym_ros.simple_rl import F110Gym

exp_no = 3
log_dir = f"logs/SAC_F1Tenth_Exp{exp_no}"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()
env.reset()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

weights_path = os.path.join(log_dir, "sac_f1tenth_final.zip")
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(weights_path):
    print(f"Loading model from {weights_path}")
    model = SAC.load(weights_path, env=env, device=device)
else:
    print("No saved model found, initializing new SAC model")
    model = SAC(
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
    )

model.set_logger(new_logger)

model.learn(
    total_timesteps=20_00_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "sac_f1tenth_final"))