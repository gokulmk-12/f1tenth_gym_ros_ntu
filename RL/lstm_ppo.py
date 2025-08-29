import os
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure

from f1tenth_gym_ros.simple_rl import F110Gym

exp_no = 1
exp_name = "algo_logs"
prev_obstacle_type, obstacle_type="no_obstacles", "no_obstacles"
prev_log_dir = f"logs/{exp_name}/LSTMPPO_F1Tenth_Exp{exp_no}/{prev_obstacle_type}1"
log_dir = f"logs/{exp_name}/LSTMPPO_F1Tenth_Exp{exp_no}/{obstacle_type}"
os.makedirs(log_dir, exist_ok=True)

env = F110Gym(is_opp=False)
env.reset()

new_logger = configure(log_dir, ["stdout", "tensorboard"])

weights_path = os.path.join(prev_log_dir, "lstmppo_f1tenth_final.zip")
device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(weights_path):
    print(f"Loading model from {weights_path}")
    model = RecurrentPPO.load(weights_path, env=env, device=device)
else:
    print("No saved model found, initializing new LSTMPPO model")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
    )

model.set_logger(new_logger)

model.learn(
    total_timesteps=500_000,
    progress_bar=True
)

model.save(os.path.join(log_dir, "lstmppo_f1tenth_final"))