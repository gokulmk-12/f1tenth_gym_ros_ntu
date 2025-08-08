import os
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

from f1tenth_gym_ros.simple_rl import F110Gym

exp_no = 1
exp_name = "obs_logs"
log_dir = f"logs/{exp_name}/A2C_F1Tenth_Exp{exp_no}"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym()
env.reset()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

weights_path = os.path.join(log_dir, "a2c_f1tenth_final.zip")
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(weights_path):
    print(f"Loading model from {weights_path}")
    model = A2C.load(weights_path, env=env, device=device)
else:
    print("No saved model found, initializing new A2C model")
    model = A2C(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        device=device,
        learning_rate=7e-4,  
        n_steps=5,           # You can increase this for more stable updates
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

model.set_logger(new_logger)

model.learn(
    total_timesteps=500_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "a2c_f1tenth_final"))
