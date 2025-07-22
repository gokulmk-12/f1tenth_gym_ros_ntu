import os
import csv
import time
import torch
import numpy as np

from stable_baselines3 import PPO
from f1tenth_gym_ros.simple_rl import F110Gym

env = F110Gym()
obs = env.reset()

model_path = "logs/PPO_F1Tenth_2/ppo_f1tenth_final"
model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

done = False
positions = []
total_reward = 0.0

log_dir = "src/f1tenth_gym_ros/race_logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"race_log_rl.csv")

with open(log_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time", "pos_x", "pos_y", "lin_vel_x"])

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    with open(log_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        time_now = time.time_ns()
        writer.writerow([time_now, env.info['pose'][0], env.info['pose'][1], env.info['speed'][0]])

    time.sleep(0.01)

print(f"Total Episode Reward: {total_reward:.2f}")