import gym
import numpy as np

map_path = "/sim_ws/src/f1tenth_gym_ros/maps/levine_closed"
map_img_ext = '.png'
env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_img_ext, num_agents=1)
obs, _, done, _ = env.reset(np.array([[0., 0., 0.]]))

while True:
    env.render()
    ego_steer, ego_requested_speed = 0.0, 0.0
    obs, reward, done, _ = env.step(np.array([[ego_steer, ego_requested_speed]]))
    if done:
        obs, _, done, _ = env.reset(np.array([[0., 0., 0.]]))