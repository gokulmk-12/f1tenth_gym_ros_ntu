import os
import csv
import time
import torch

from stable_baselines3 import PPO, SAC
from f1tenth_gym_ros.simple_rl import F110Gym

env = F110Gym()
obs = env.reset()

exp_no = 3
algo = "PPO"
model_path = f"logs/{algo}_F1Tenth_Exp{exp_no}/{algo.lower()}_f1tenth_final"
if algo == "PPO":
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
elif algo == "SAC":
    model = SAC.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

done = False
positions = []
total_reward = 0.0

log_dir = "race_logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"race_log_rl_levine_ppo_2.csv")

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