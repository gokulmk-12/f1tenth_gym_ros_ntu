#!/usr/bin/env python3

import csv
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher')

        self.marker_pub = self.create_publisher(Marker, 'waypoints', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.declare_parameter("map_name", "Spielberg")  # Default: Spielberg
        map_name = self.get_parameter("map_name").get_parameter_value().string_value

        self.csv_file = f"/sim_ws/src/f1tenth_gym_ros/tracks/{map_name}_mincurv.csv"
        self.config = f"/sim_ws/src/f1tenth_gym_ros/maps/{map_name}.yaml"
        self.points = self.load_points_from_csv(self.csv_file)
    
    def pixel_to_map(self, px, py, resolution, origin, imag_height):
        map_x = origin[0] + px * resolution
        map_y = origin[1] + (imag_height - py) * resolution
        return map_x, map_y
    
    def load_points_from_csv(self, file_path):
        points = []

        with open(self.config, 'r') as yaml_file:
            try:
                self.map_metadata = yaml.safe_load(yaml_file)
                self.map_resolution = self.map_metadata['resolution']
                self.origin = self.map_metadata['origin']
            except yaml.YAMLError as e:
                print(e)

        try:
            with open(file_path, 'r') as csvfile:
                file = csv.reader(csvfile)
                for row in file:
                    x, y = float(row[0]), float(row[1])
                    points.append((x, y))
        except Exception as e:
            self.get_logger().error(f"Error reading CSV: {e}")

        return points

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        marker.scale.x, marker.scale.y, marker.scale.z = 0.08, 0.08, 0.08
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.8, 0.1, 1.0

        marker.points = [Point(x=pt[0], y=pt[1], z=0.0) for pt in self.points]

        self.marker_pub.publish(marker)
        self.get_logger().info(f"Published {len(marker.points)} points as markers.")
    
def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()