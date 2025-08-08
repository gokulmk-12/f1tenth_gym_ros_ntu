import gym
import csv
import math
import yaml
import rclpy
import gym.spaces
import numpy as np
from PIL import Image
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from threading import Event

class F110Gym(gym.Env):
    def __init__(self, config_file="sim.yaml", is_train=True, is_opp=False):
        super(F110Gym, self).__init__()

        rclpy.init()
        self.node = rclpy.create_node('f110_rl_node')

        config = f"/sim_ws/src/f1tenth_gym_ros/config/{config_file}"
        self.config = yaml.safe_load(open(config, 'r'))
        self.config = self.config['bridge']['ros__parameters']
        self.opp = is_opp

        self.lidar_dim = self.config['reduce_lidar_data']
        self.laser = np.zeros(self.lidar_dim)
        self.speed = np.zeros(3)
        self.lookAheadDis = 2.0
        self.wheel_base = 0.3302
        self.Kp = 1.0
        self.flag, self.flag_opp = False, False
        self.is_train = is_train

        self.action_space = gym.spaces.Box(
            low=np.array([
                self.config['steering_min'],
                self.config['speed_min'],
            ]),
            high=np.array([
                self.config['steering_max'], 
                self.config['speed_max'],
            ]),
            dtype=np.float64
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lidar_dim*2+103, ), dtype=np.float64
        )

        self.drive_pub = self.node.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.reset_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.laser_event, self.ego_odom_event, self.opp_odom_event = Event(), Event(), Event()
        self.info = {}
        map_name = str(self.config['map_name'])
        self.csv_file = f"/sim_ws/src/f1tenth_gym_ros/tracks/{map_name}.csv"
        self.img_file = f"/sim_ws/src/f1tenth_gym_ros/maps/{map_name}.png"
        self.map_yaml_file = f"/sim_ws/src/f1tenth_gym_ros/maps/{map_name}.yaml"
        self.points = self.load_points_from_csv(self.csv_file)
        self.points_opp = self.load_points_from_csv(f"/sim_ws/src/f1tenth_gym_ros/tracks/{map_name}.csv")
        self.map_img = np.array(Image.open(self.img_file).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.cut_region = 30

        self.map_config = yaml.safe_load(open(self.map_yaml_file, 'r'))
        self.map_resolution = self.map_config['resolution']
        self.map_origin = self.map_config['origin']

        self.node.create_subscription(Odometry, 'ego_racecar/odom', self.ego_odom_callback, 10)
        self.node.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        if self.opp:
            self.node.create_subscription(Odometry, 'opp_racecar/odom', self.opp_odom_callback, 10)
            self.opp_drive_pub = self.node.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.steps_until_next_pose = 0
    
    def load_points_from_csv(self, file_path):
        points = []
        try:
            with open(file_path, 'r') as csvfile:
                file = csv.reader(csvfile)
                for row in file:
                    x, y, v = float(row[0]), float(row[1]), float(row[2])
                    points.append((x, y, v))
        except Exception as e:
            self.node.get_logger().error(f"Error reading CSV: {e}")

        return points

    def ego_odom_callback(self, msg):
        try:
            self.current_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.current_heading = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
            
            self.pose = np.array([self.current_pos[0], self.current_pos[1], self.current_heading])
            self.speed = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
            self.info["pose"] = self.pose
            self.info["speed"] = self.speed
            self.ego_odom_event.set()
        except Exception as e:
            print(f"Failed to get current car pose, taking previous: {e}")
    
    def opp_odom_callback(self, msg):
        try:
            self.current_pos_opp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.current_heading_opp = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
            
            self.pose_opp = np.array([self.current_pos[0], self.current_pos[1], self.current_heading])
            self.speed_opp = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
            self.opp_odom_event.set()
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

    def _get_obs(self, info, speed, goalPt):
        map_img = self.get_map_region()
        obs = np.hstack([
                    info["last_scan"],      # Previous Lidar Scan
                    info["current_scan"],   # Current Lidar Scan
                    np.array(speed),        # Current Speed
                    np.array(goalPt[2]),    # Min Time Speed
                    map_img,                # Cropped Image of Map around car
                ])
        return obs

    def pt_to_pt_distance(self, pt1, pt2):
        dist = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
        return dist
    
    def run_pp(self, dlookahead=0):
        currentX, currentY = self.current_pos[0], self.current_pos[1] 
        
        if not self.flag:
            shortest_dist = np.inf
            for i in range(0, len(self.points)):
                if self.pt_to_pt_distance(self.points[i], self.current_pos) < shortest_dist:
                    shortest_dist = self.pt_to_pt_distance(self.points[i], self.current_pos)
                    self.lastFoundIndex = i
            self.flag = True
        
        while self.pt_to_pt_distance(self.points[self.lastFoundIndex], self.current_pos) < (self.lookAheadDis + dlookahead):
            self.lastFoundIndex += 1
            if (self.lastFoundIndex > len(self.points) - 1):
                self.lastFoundIndex = 0
        
        goalPt = self.points[self.lastFoundIndex]
        dx = goalPt[0] - currentX
        dy = goalPt[1] - currentY
        goal_heading = math.atan2(dy, dx)
        
        sin_alpha = math.sin(goal_heading - self.current_heading)
        angle = self.Kp * (np.arctan(2.0 * self.wheel_base * sin_alpha)) / (self.lookAheadDis + dlookahead)
        return goalPt, angle

    def run_pp_opp(self, dlookahead=0):
        currentX, currentY = self.current_pos_opp[0], self.current_pos_opp[1] 
        
        if not self.flag_opp:
            shortest_dist = np.inf
            for i in range(0, len(self.points_opp)):
                if self.pt_to_pt_distance(self.points_opp[i], self.current_pos_opp) < shortest_dist:
                    shortest_dist = self.pt_to_pt_distance(self.points_opp[i], self.current_pos_opp)
                    self.lastFoundIndex_opp = i
            self.flag_opp = True
        
        while self.pt_to_pt_distance(self.points_opp[self.lastFoundIndex_opp], self.current_pos_opp) < (self.lookAheadDis + dlookahead):
            self.lastFoundIndex_opp += 1
            if (self.lastFoundIndex_opp > len(self.points_opp) - 1):
                self.lastFoundIndex_opp = 0
        
        goalPt = self.points_opp[self.lastFoundIndex_opp]
        dx = goalPt[0] - currentX
        dy = goalPt[1] - currentY
        goal_heading = math.atan2(dy, dx)
        
        sin_alpha = math.sin(goal_heading - self.current_heading_opp)
        angle = self.Kp * (np.arctan(2.0 * self.wheel_base * sin_alpha)) / (self.lookAheadDis + dlookahead)
        return goalPt, angle
    
    
    def _sample_new_pose(self):
        new_pose = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            # [5.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
            [9.75, 4.63, 0.0, 0.0, 0.7, 0.714],
            # [5.73, 8.86, 0.0, 0.0, 0.9999, 0.009],
            [-0.107, 8.97, 0.0, 0.0, 0.9999, 0.009],
            [-8.768, 8.697, 0.0, 0.0, 0.9998, -0.017],
            # [-13.768, 4.3916, 0.0, 0.0, 0.7103, -0.704]
        ]
        idx = np.random.randint(low=0, high=len(new_pose))
        print(f"Setting New Pose: {new_pose[idx][0], new_pose[idx][1]}")
        _, _, yaw = euler_from_quaternion([0.0, 0.0, new_pose[idx][4], new_pose[idx][5]])
        self.config['sx'], self.config['sy'], self.config['stheta'] = new_pose[idx][0], new_pose[idx][1], yaw

    def get_map_region(self):
        current_x, current_y = (self.current_pos[0] - self.map_origin[0]) / self.map_resolution, (self.current_pos[1] - self.map_origin[1]) / self.map_resolution
        left, right = current_x - self.cut_region, current_x + self.cut_region
        upper, lower = current_y - self.cut_region, current_y + self.cut_region

        map_img = Image.fromarray(self.map_img)
        cropped_img = np.array(map_img.crop((left, upper, right, lower)))

        H, W = cropped_img.shape
        cropped_img = cropped_img.reshape(H // 6, 6, W // 6, 6). max(axis=(1, 3))
        return cropped_img.flatten()
    
    def reset(self):
        msg = PoseWithCovarianceStamped()
        sx, sy, yaw = self.config['sx'], self.config['sy'], self.config['stheta']
        # _, _, yaw = euler_from_quaternion([0.0, 0.0, self.config['ori_z'], self.config['ori_w']])
        _, _, z, w = quaternion_from_euler(0., 0., yaw)
        self.pose = np.array([sx, sy, yaw])
        msg.pose.pose.position.x = sx
        msg.pose.pose.position.y = sy
        msg.pose.pose.orientation.z = z
        msg.pose.pose.orientation.w = w
        self.reset_pub.publish(msg)

        drive = AckermannDriveStamped()
        self.speed = np.array([0.0, 0.0])
        drive.drive.speed = 0.0
        drive.drive.steering_angle = 0.0
        self.drive_pub.publish(drive)

        self.laser_event.clear()
        self.ego_odom_event.clear()
        if self.opp:
            self.opp_odom_event.clear()
        self.action = [0., 0.]

        while not (self.laser_event.is_set() and self.ego_odom_event.is_set()):
            if self.opp:
                while not self.opp_odom_event.is_set():
                    rclpy.spin_once(self.node, timeout_sec=0.01)
            else:
                rclpy.spin_once(self.node, timeout_sec=0.01)
                
        self.info = {
            "step": 0,
            "pose": self.pose,
            "speed": self.speed,
            "last_scan": self.laser.copy(),
            "current_scan": self.laser.copy(),
            "last_action": self.action.copy(),
            "current_action": self.action.copy(),
        }

        self.flag = False
        obs = self._get_obs(self.info, self.speed, [0., 0., 0.])
        return obs
    
    def step(self, action):
        self.laser_event.clear()
        self.ego_odom_event.clear()
        if self.opp:
            self.opp_odom_event.clear()

        goalPt, angle = self.run_pp()
        lat_forceweight = np.sqrt((0.015 * 9.81 * (self.lookAheadDis))/ np.tan(abs(angle)))
        base_speed = min(2.0, lat_forceweight)
        final_speed = base_speed * (1 + action[1])

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = float(angle+action[0])
        drive.drive.speed = float(final_speed)
        self.drive_pub.publish(drive)

        if self.opp:
            _, angle_opp = self.run_pp_opp()
            lat_forceweight_opp = np.sqrt((0.015 * 9.81 * (self.lookAheadDis))/ np.tan(abs(angle_opp)))
            base_speed_opp = min(2.0, lat_forceweight_opp)

            drive_opp = AckermannDriveStamped()
            drive_opp.drive.steering_angle = float(angle_opp)
            drive_opp.drive.speed = float(base_speed_opp)
            self.opp_drive_pub.publish(drive_opp)

        while not (self.laser_event.is_set() and self.ego_odom_event.is_set()):
            if self.opp:
                while not self.opp_odom_event.is_set():
                    rclpy.spin_once(self.node, timeout_sec=0.01)
            else:
                rclpy.spin_once(self.node, timeout_sec=0.01)
        
        self.info["step"] += 1
        self.steps_until_next_pose += 1
        self.info["last_action"] = self.info["current_action"]
        self.info["current_action"] = np.array(action)

        if self.steps_until_next_pose % 10000 == 0:
            self._sample_new_pose()
        
        obs = self._get_obs(self.info, self.speed, goalPt)
        done = self._get_termination(self.info)

        if self.info["step"] > 1000 and self.is_train:
            print("Max episode length reached. Forcing reset.")
            done = True

        reward = sum(self._get_reward(self.info, done).values())
        return obs, reward, done, {}
    
    def _get_reward(self, info, done):
        return {
            'linvel_reward': info["speed"][0] * self.config['vel_norm'] * self.config['vel_reward_scale'],
            'current_laser_reward': min(list(info['current_scan'])) * self.config['current_lidar_reward_scale'],
            'termination': -20 if done else 0,
            'smoothness': -self.config["smoothness_scale"] * np.sum(np.square(self.info["current_action"] - self.info["last_action"])),
        }
    
    def _get_termination(self, info):
        current_scan_termination = np.min(info['current_scan']) < 0.013
        previous_scan_termination = np.min(info['last_scan']) < 0.015
        return current_scan_termination
    
    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    
    sim = F110Gym(is_opp=True)
    obs = sim.reset()

    while True:
        action = np.zeros(2)
        obs, reward, done, _ = sim.step(action)
        if done:
            obs = sim.reset()
    
    sim.close()