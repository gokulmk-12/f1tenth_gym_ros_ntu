#!/usr/bin/env python3

import csv
import math
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion

class SimplePurePursuit(Node):
    def __init__(self):
        super().__init__("simple_pp")
        self.lookAheadDis = 2.1
        self.wheel_base = 0.3302
        self.Kp = 1.0
        self.flag = False
        # self.waypoint_sub = self.create_subscription(Marker, 'waypoints', self.waypoint_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.marker_pub = self.create_publisher(Marker, 'start', 10)
        self.ack_drive_pub = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

        map_name = "levine_closed"
        self.csv_file = f"/sim_ws/src/f1tenth_gym_ros/tracks/{map_name}.csv"
        self.path = self.load_points_from_csv(self.csv_file)
    
    def load_points_from_csv(self, file_path):
        points = []
        try:
            with open(file_path, 'r') as csvfile:
                file = csv.reader(csvfile)
                for row in file:
                    x, y, v = float(row[0]), float(row[1]), float(row[2])
                    points.append((x, y, v))
        except Exception as e:
            self.get_logger().error(f"Error reading CSV: {e}")

        return points
    
    def odom_callback(self, msg):
        try:
            # self.get_logger().info(f"Recieved current car pose")
            self.current_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.current_heading = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        except Exception as e:
            print(f"Failed to get current car pose, taking previous: {e}")
    
    def scan_callback(self, msg):
        self.range_min, self.range_max = msg.range_min, msg.range_max
        self.angle_min, self.angle_max = msg.angle_min, msg.angle_max
        angles = np.round(np.linspace(self.angle_min, self.angle_max, num=len(msg.ranges)), 2)
        
        req_angle_min, req_angle_max = np.deg2rad(-90), np.deg2rad(90)
        mask = (angles >= req_angle_min) & (angles <= req_angle_max)
        scan = np.array(msg.ranges)[mask]
        scan = scan[np.isfinite(scan)]
            
    def timer_callback(self):
        goalPt, angle = self.run_pp()

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.scale.x, marker.scale.y, marker.scale.z = 0.1, 0.1, 0.1
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.8, 0.1, 0.0, 1.0

        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = goalPt[0], goalPt[1], 0.0
        self.marker_pub.publish(marker)

        drive = AckermannDriveStamped()
        drive.drive.steering_angle = angle

        lat_forceweight = np.sqrt((0.015 * 9.81 * self.lookAheadDis)/ np.tan(abs(angle)))
        speed = min(2.0, lat_forceweight)

        # if (abs(angle) > 20.0 / 180.0 * np.pi):
        #     speed = 0.5
        # elif (abs(angle) > 10.0 / 180.0 * np.pi):
        #     speed = 1.5
        # else:
        #     speed = 2.5
        
        drive.drive.speed = speed 
            
        self.ack_drive_pub.publish(drive)
        self.get_logger().info(f"Vehicle Speed: {angle, speed, self.lookAheadDis}")
    
    def pt_to_pt_distance(self, pt1, pt2):
        dist = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
        return dist
    
    def run_pp(self):
        currentX, currentY = self.current_pos[0], self.current_pos[1]

        if not self.flag:
            shortest_dist = np.inf
            for i in range(0, len(self.path)):
                if self.pt_to_pt_distance(self.path[i], self.current_pos) < shortest_dist:
                    shortest_dist = self.pt_to_pt_distance(self.path[i], self.current_pos)
                    self.lastFoundIndex = i
            self.flag = True
        
        while self.pt_to_pt_distance(self.path[self.lastFoundIndex], self.current_pos) < self.lookAheadDis:
            self.lastFoundIndex += 1
            if (self.lastFoundIndex > len(self.path) - 1):
                self.lastFoundIndex = 0
        
        goalPt = self.path[self.lastFoundIndex]
        dx = goalPt[0] - currentX
        dy = goalPt[1] - currentY
        goal_heading = math.atan2(dy, dx)
        
        sin_alpha = math.sin(goal_heading - self.current_heading)
        angle = self.Kp * (np.arctan(2.0 * self.wheel_base * sin_alpha)) / self.lookAheadDis
        return goalPt, angle
        _
def main(args=None):
    rclpy.init(args=args)
    node = SimplePurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()