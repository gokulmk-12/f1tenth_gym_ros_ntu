import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from f1tenth_gym_ros.simple_rl import F110Gym

exp_no = 1
exp_name = "obs_logs"
log_dir = f"logs/{exp_name}/PPO_F1Tenth_Exp{exp_no}"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()
env.reset()

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
        learning_rate=3e-4,
        batch_size=256,
        ent_coef=0.02,
        max_grad_norm=1.0,
        use_sde=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
model.set_logger(new_logger)

model.learn(
    total_timesteps=500_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "ppo_f1tenth_final"))