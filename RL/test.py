import os
import csv
import time
import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise

from f1tenth_gym_ros.simple_rl import F110Gym

env = F110Gym(is_train=False, is_opp=False)
obs = env.reset()

exp_no = 1
algo = "SAC"
logs_type = "obs_logs"
obstacle_type="no_obstacles"
model_path = f"logs/{logs_type}/{algo}_F1Tenth_Exp{exp_no}/{obstacle_type}/{algo.lower()}_f1tenth_final"
# model_path = f"logs/{logs_type}/{algo}_F1Tenth_Exp{exp_no}/{algo.lower()}_f1tenth_final"
device = "cuda" if torch.cuda.is_available() else "cpu"

if algo == "PPO":
    model = PPO.load(model_path, env=env, device=device)
elif algo == "A2C":
    model = A2C.load(model_path, env=env, device=device)
elif algo == "SAC":
    model = SAC.load(model_path, env=env, device=device)
elif algo == "TD3":
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3.load(model_path, env=env, device=device)
    model.action_noise = action_noise

done = False
positions = []
total_reward = 0.0

write_to_file = False

if write_to_file:
    log_dir, map_name = "race_logs", "levine_closed"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"race_log_rl_{map_name}_{algo.lower()}_mapimg.csv")

    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "pos_x", "pos_y", "lin_vel_x"])

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    if write_to_file:
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            time_now = time.time_ns()
            writer.writerow([time_now, env.info['pose'][0], env.info['pose'][1], env.info['speed'][0]])

    time.sleep(0.01)

print(f"Total Episode Reward: {total_reward:.2f}")