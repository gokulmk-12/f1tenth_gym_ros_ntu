import os
import gym
import yaml
import rclpy
import gym.spaces
import numpy as np
import pandas as pd
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped

from threading import Event

class F110Gym(gym.Env):
    def __init__(self, config_file="sim.yaml"):
        super(F110Gym, self).__init__()

        rclpy.init()
        self.node = rclpy.create_node('f110_rl_node')

        config = f"/sim_ws/src/f1tenth_gym_ros/config/{config_file}"
        self.config = yaml.safe_load(open(config, 'r'))
        self.config = self.config['bridge']['ros__parameters']

        self.lidar_dim = self.config['reduce_lidar_data']
        self.laser = np.zeros(self.lidar_dim)
        self.speed = np.zeros(3)

        self.action_space = gym.spaces.Box(
            low=np.array([
                self.config['steering_min'], 
                self.config['speed_min']
            ]),
            high=np.array([
                self.config['steering_max'], 
                self.config['speed_max']
            ]),
            dtype=np.float64
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lidar_dim*2+2, ), dtype=np.float64
        )

        self.drive_pub = self.node.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.reset_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.laser_event, self.odom_event = Event(), Event()
        self.info = {}

        self.node.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.node.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.steps_until_next_pose = 0

    def odom_callback(self, msg):
        try:
            self.current_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.current_heading = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
            
            self.pose = np.array([self.current_pos[0], self.current_pos[1], self.current_heading])
            self.speed = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
            self.info["pose"] = self.pose
            self.info["speed"] = self.speed
            self.odom_event.set()
        except Exception as e:
            print(f"Failed to get current car pose, taking previous: {e}")
    
    def scan_callback(self, msg):
        scan = msg.ranges[180: -180]
        self.laser = np.array(self.process_lidar(scan))

        self.info['last_scan'] = self.info.get('current_scan', self.laser)
        self.info['current_scan'] = self.laser
        
        self.laser_event.set()

    def process_lidar(self, scan):
        max_distance= self.config['max_distance_norm']
        reduce_by = self.config['reduce_lidar_data']

        if self.config['lidar_reduction_method'] == 'avg':
            lidar_avg = []
            for i in range(0, len(scan), reduce_by):
                segment = list(filter(lambda x:  x <= max_distance, scan[i:i + reduce_by]))
                if segment:
                    lidar_avg.append(sum(segment)/len(segment))
                else:
                    lidar_avg.append(max_distance)
        else:
            lidar_avg = scan
        
        if max_distance > 1:
            lidar_avg = [x / max_distance for x in lidar_avg]
        
        return lidar_avg

    def _get_obs(self, info, speed):
        obs = np.concatenate([
                    info["last_scan"],
                    info["current_scan"], 
                    np.array(speed),
                ])
        return obs
    
    def _sample_new_pose(self):
        new_pose = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [5.1, 0.0, 0.0, 0.0, 0.0, 1.0], 
            [9.75, 4.63, 0.0, 0.0, 0.7, 0.714],
            [-0.84, 8.873, 0.0, 0.0, 0.9997, 0.02419],
            [-13.85, 6.105, 0.0, 0.0, 0.6077, -0.7941]
        ]
        idx = np.random.randint(low=0, high=len(new_pose))
        print(f"Setting New Pose: {new_pose[idx][0], new_pose[idx][1]}")
        self.config['sx'], self.config['sy'], self.config['ori_z'], self.config['ori_w'] = new_pose[idx][0], new_pose[idx][1], new_pose[idx][4], new_pose[idx][5]
    
    def reset(self):
        msg = PoseWithCovarianceStamped()
        sx, sy, yaw = self.config['sx'], self.config['sy'], 0.0
        _, _, yaw = euler_from_quaternion([0.0, 0.0, self.config['ori_z'], self.config['ori_w']])
        self.pose = np.array([sx, sy, yaw])
        msg.pose.pose.position.x = sx
        msg.pose.pose.position.y = sy
        msg.pose.pose.orientation.z = self.config['ori_z']
        msg.pose.pose.orientation.w = self.config['ori_w']
        self.reset_pub.publish(msg)

        drive = AckermannDriveStamped()
        self.speed = np.array([0.0, 0.0])
        drive.drive.speed = 0.0
        drive.drive.steering_angle = 0.0
        self.drive_pub.publish(drive)

        self.laser_event.clear()
        self.odom_event.clear()

        while not (self.laser_event.is_set() and self.odom_event.is_set()):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        self.info = {
            "step": 0,
            "pose": self.pose,
            "speed": self.speed,
            "last_scan": self.laser.copy(),
            "current_scan": self.laser.copy(),
        }

        obs = self._get_obs(self.info, self.speed)
        return obs
    
    def step(self, action):
        self.laser_event.clear()
        self.odom_event.clear()

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = float(action[0])
        drive.drive.speed = float(action[1])
        self.drive_pub.publish(drive)

        while not (self.laser_event.is_set() and self.odom_event.is_set()):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        self.info["step"] += 1
        self.steps_until_next_pose += 1

        if self.steps_until_next_pose % 100000 == 0:
            self._sample_new_pose()
        
        obs = self._get_obs(self.info, self.speed)
        done = self._get_termination(self.info)
        rewards = self._get_reward(self.info, done)
        reward = np.clip(sum(rewards.values()), -1, 1)

        return obs, reward, done, {}
    
    def _get_reward(self, info, done):
        return {
            'linvel_reward': info["speed"][0] * self.config['vel_norm'] * self.config['vel_reward_scale'],
            'current_laser_reward': min(list(info['current_scan'])) * self.config['current_lidar_reward_scale'],
            'previous_laser_reward': min(list(info['last_scan'])) * self.config['previous_lidar_reward_scale'],
            'termination': -1 if done else 0
        }
    
    def _get_termination(self, info):
        current_scan_termination = np.min(info['current_scan']) < 0.015
        previous_scan_termination = np.min(info['last_scan']) < 0.015
        return current_scan_termination
    
    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    
    sim = F110Gym()
    obs = sim.reset()

    while True:
        action = sim.action_space.sample()
        obs, reward, done, _ = sim.step(action)
        if done:
            obs = sim.reset()
    
    sim.close()