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
        self.is_train = is_train

        self.lidar_dim = self.config['reduce_lidar_data']
        self.laser = np.zeros(self.lidar_dim)
        self.speed = np.zeros(3)
        self.lookAheadDis = 2.0
        self.wheel_base = 0.3302
        self.Kp = 1.0
        self.flag, self.flag_opp = False, False

        ## Gym Action Space Definition
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

        ## Gym Observation Space Definition
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.lidar_dim*2+3, ), dtype=np.float64
            ## Current LiDAR Scan, Previous LiDAR Scan, Linear Vel X, Ang Vel Z, Min Time Vel X
        )

        ## Publisher Definitions
        self.drive_pub = self.node.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.reset_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.laser_event, self.ego_odom_event, self.opp_odom_event = Event(), Event(), Event()
        self.info = {}

        map_name = str(self.config['map_name'])
        self.csv_file = f"/sim_ws/src/f1tenth_gym_ros/tracks/{map_name}.csv"
        self.img_file = f"/sim_ws/src/f1tenth_gym_ros/maps/{map_name}.png"
        self.map_yaml_file = f"/sim_ws/src/f1tenth_gym_ros/maps/{map_name}.yaml"
        self.points = self.load_points_from_csv(self.csv_file)
        if is_opp:
            self.points_opp = self.load_points_from_csv(f"/sim_ws/src/f1tenth_gym_ros/tracks/{map_name}_opp_ways.csv")
        self.map_img = np.array(Image.open(self.img_file).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.cut_region = 30

        self.map_config = yaml.safe_load(open(self.map_yaml_file, 'r'))
        self.map_resolution = self.map_config['resolution']
        self.map_origin = self.map_config['origin']

        ## Subscriber Definitions
        self.node.create_subscription(Odometry, 'ego_racecar/odom', self.ego_odom_callback, 10)
        self.node.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        ## Publisher and Subscriber for Opp Racecar if present
        if self.opp:
            self.node.create_subscription(Odometry, 'opp_racecar/odom', self.opp_odom_callback, 10)
            self.opp_drive_pub = self.node.create_publisher(AckermannDriveStamped, '/opp_drive', 10)
        self.steps_until_next_pose = 0

        self.forward_arc_deg = 30
        self.deg_per_beam = 270. / self.lidar_dim
        half_beams = int((self.forward_arc_deg / self.deg_per_beam) / 2)
        self.forward_center = self.lidar_dim // 2
        self.forward_slice = slice(self.forward_center - half_beams, self.forward_center + half_beams +1)
    
    def load_points_from_csv(self, file_path):
        '''
        Purpose: Load waypoints from CSV
        Input: File path
        Output: Bunch of Waypoints
        '''
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
        '''
        Purpose: Subscriber callback for ego racecar odometry
        Input: ros2 odometry msg
        Output: None
        '''
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
        '''
        Purpose: Subscriber callback for opp racecar odometry
        Input: ros2 odometry msg
        Output: None
        Active only when self.is_opp is set True
        '''
        try:
            self.current_pos_opp = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.current_heading_opp = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
            
            self.pose_opp = np.array([self.current_pos[0], self.current_pos[1], self.current_heading])
            self.speed_opp = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
            self.opp_odom_event.set()
        except Exception as e:
            print(f"Failed to get current car pose, taking previous: {e}")
    
    def scan_callback(self, msg):
        '''
        Purpose: Subscriber callback for ego racecar laserscan
        Input: ros2 laserscan msg
        Output: None
        Run as an event 
        '''
        scan = msg.ranges[180: -180]
        self.laser = np.array(self.process_lidar(scan))

        self.info['last_scan'] = self.info.get('current_scan', self.laser)
        self.info['current_scan'] = self.laser

        front_scan = self.info['current_scan'][self.forward_slice]
        front_min_idx_local = int(np.argmin(front_scan))
        front_min = float(front_scan[front_min_idx_local])
        self.info["front_min"] = front_min
        
        self.laser_event.set()

    def process_lidar(self, scan):
        '''
        Purpose: Process 1080 laserscans to lidar_dim using averaging method
        Input: list of scans
        Output: averaged scan data
        Visit sim.yaml for more info 
        '''
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

    def _get_observations(self, info, speed, goalPt):
        '''
        Purpose: Gym RL style get observation function
        Input: info dict, current speed of racecar, next goalPt
        Output: array of observations for RL
        '''
        map_img = self.get_map_region()
        obs = np.hstack([
                    info["last_scan"],      # Previous Lidar Scan
                    info["current_scan"],   # Current Lidar Scan
                    np.array(speed),        # Current Speed
                    np.array(goalPt[2]),    # Min Time Speed
                ])
        return obs

    def pt_to_pt_distance(self, pt1, pt2):
        '''
        Purpose: Find point-point euclidean distance
        Input: 2 points
        Output: euclidean distance
        '''
        dist = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
        return dist
    
    def run_pp(self, dlookahead=0):
        '''
        Purpose: Simple Pure Pursuit Algorithm for ego racecar
        Input: None
        Output: Next goalPt, Steering angle
        '''
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
        '''
        Purpose: Simple Pure Pursuit Algorithm for opp racecar
        Input: None
        Output: Next goalPt, Steering angle
        '''
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
        '''
        Purpose: Function to reset the agent to different start points, for generalization
        Input: None
        Output: Overwrite start position of the agent
        '''
        new_pose = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [9.75, 4.63, 0.0, 0.0, 0.7, 0.714],
            [-0.107, 8.97, 0.0, 0.0, 0.9999, 0.009],
            # [-8.768, 8.697, 0.0, 0.0, 0.9998, -0.017],
        ]
        idx = np.random.randint(low=0, high=len(new_pose))
        print(f"Setting New Pose: {new_pose[idx][0], new_pose[idx][1]}")
        _, _, yaw = euler_from_quaternion([0.0, 0.0, new_pose[idx][4], new_pose[idx][5]])
        self.config['sx'], self.config['sy'], self.config['stheta'] = new_pose[idx][0], new_pose[idx][1], yaw

    def get_map_region(self):
        '''
        Purpose: Crop out map region around the current racecar location, for observation
        Input: None
        Output: Cut-Map image
        '''
        current_x, current_y = (self.current_pos[0] - self.map_origin[0]) / self.map_resolution, (self.current_pos[1] - self.map_origin[1]) / self.map_resolution
        left, right = current_x - self.cut_region, current_x + self.cut_region
        upper, lower = current_y - self.cut_region, current_y + self.cut_region

        map_img = Image.fromarray(self.map_img)
        cropped_img = np.array(map_img.crop((left, upper, right, lower)))

        H, W = cropped_img.shape
        cropped_img = cropped_img.reshape(H // 6, 6, W // 6, 6). max(axis=(1, 3))
        return cropped_img.flatten()
    
    def reset(self):
        '''
        Purpose: Gym-RL style reset the agent, set the car speed to 0, clear the events
        Input: None
        Output: Current observation
        '''
        msg = PoseWithCovarianceStamped()
        sx, sy, yaw = self.config['sx'], self.config['sy'], self.config['stheta']
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
        obs = self._get_observations(self.info, self.speed, [0., 0., 0.])
        return obs
    
    def step(self, action):
        '''
        Purpose: Gym-RL style step forward the agent one time-step
        Input: action from policy
        Output: observation, reward, done, info
        '''
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
        
        obs = self._get_observations(self.info, self.speed, goalPt)
        done = self._get_termination(self.info)

        ### NOTE: Enable max step length only when training with Off-Policy RL algorithms
        if self.info["step"] > 1000 and self.is_train:
            print("Max episode length reached. Forcing reset.")
            done = True

        reward = sum(self._get_reward(self.info, done, goalPt).values())
        return obs, reward, done, {}
    
    def _get_reward(self, info, done, goalPt):
        '''
        Purpose: Gym-RL style reward function
        Input: info dict and done
        Output: dict of rewards
        '''
        d_clear = self.config.get("clear_path_dist", 1.0)
        v = float(self.info["speed"][0])
        v_opt = float(goalPt[2])

        if self.info["front_min"] > d_clear:
            v_track = -self.config.get("v_track_scale", 0.5) * (v - v_opt)**2
            headway_term = 0.0
        else:
            gap = self.info["front_min"]
            safe = self.config.get("safe_gap", 3.0)
            headway_term = - self.config.get("headway_scale", 0.2) * max(0.0, safe - gap)
            v_track = - self.config.get("v_track_close_scale", 0.1) * (v - min(v_opt, 2.0))**2

        return {
            'linvel_reward': info["speed"][0] * self.config['vel_norm'] * self.config['vel_reward_scale'],
            'current_laser_reward': min(list(info['current_scan'])) * self.config['current_lidar_reward_scale'],
            'termination': -20 if done else 0,
            'smoothness': -self.config["smoothness_scale"] * np.sum(np.square(self.info["current_action"] - self.info["last_action"])),
            'speed_track': v_track,
            'headway': headway_term
        }
    
    def _get_termination(self, info):
        '''
        Purpose: Gym-RL style for terminating the agent earlier
        Input: info dict
        Output: termination condition
        '''
        current_scan_termination = np.min(info['current_scan']) < 0.013
        previous_scan_termination = np.min(info['last_scan']) < 0.015
        return current_scan_termination
    
    def close(self):
        '''
        Purpose: Destroy the ROS2 Node and shutdown rclpy
        Input: None
        Output: None
        '''
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