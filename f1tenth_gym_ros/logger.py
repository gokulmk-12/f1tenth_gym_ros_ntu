import os
import csv
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class Logger(Node):
    def __init__(self):
        super(Logger, self).__init__(node_name="race_logger")

        self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        
        log_dir = "src/f1tenth_gym_ros/race_logs"
        os.makedirs(log_dir, exist_ok=True)
        map_name = "levine_closed"
        self.log_file_path = os.path.join(log_dir, f"race_log_{map_name}_pp.csv")

        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "pos_x", "pos_y", "lin_vel_x", "ang_vel_z"])
        
        self.get_logger().info(f"Logging to {self.log_file_path}")

        self.last_pos = None
        self.lap_start_time = None
        self.lap_times = []

    def odom_callback(self, msg):
        try:
            pos_x = msg.pose.pose.position.x
            pos_y = msg.pose.pose.position.y
            lin_vel_x = msg.twist.twist.linear.x
            ang_vel_z = msg.twist.twist.angular.z
            time_now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9

            with open(self.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time_now, pos_x, pos_y, lin_vel_x, ang_vel_z])

            curr_pos = (pos_x, pos_y)
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if self.last_pos is not None:
                if self.crossed_line(self.last_pos, curr_pos, line_x=0.0):
                    if self.lap_start_time is not None:
                        lap_time = now_sec - self.lap_start_time
                        self.lap_times.append(lap_time)
                        self.get_logger().info(f"Lap completed in {lap_time:.2f} seconds")
                    self.lap_start_time = now_sec
            self.last_pos = curr_pos

        except Exception as e:
            self.get_logger().error(f"Failed to log odometry: {e}") 
    
    def crossed_line(self, last_pos, curr_pos, line_x=0.0):
        return last_pos[0] < line_x and curr_pos[0] >= line_x

def main(args=None):
    rclpy.init(args=args)
    node = Logger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
