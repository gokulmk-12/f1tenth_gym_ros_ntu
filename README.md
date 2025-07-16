# F1TENTH gym environment ROS2 communication bridge
This is a containerized ROS communication bridge for the F1TENTH gym environment that turns it into a simulation in ROS2.

# Installation

## Native on Ubuntu 20.04

**Install the following dependencies:**
- **ROS 2** Follow the instructions [here](https://docs.ros.org/en/foxy/Installation.html) to install ROS 2 Foxy.
- **F1TENTH Gym**
  ```bash
  git clone https://github.com/f1tenth/f1tenth_gym
  cd f1tenth_gym && pip3 install -e .
  ```

**Installing the simulation:**
- Create a workspace: ```cd $HOME && mkdir -p sim_ws/src```
- Clone the repo into the workspace:
  ```bash
  cd $HOME/sim_ws/src
  git clone https://github.com/f1tenth/f1tenth_gym_ros_ntu
  ```
- Update correct parameter for path to map file:
  Go to `sim.yaml` [https://github.com/f1tenth/f1tenth_gym_ros_ntu/blob/main/config/sim.yaml](https://github.com/f1tenth/f1tenth_gym_ros/blob/main/config/sim.yaml) in your cloned repo, change the `map_path` parameter to point to the correct location. It should be `'<your_home_dir>/sim_ws/src/f1tenth_gym_ros_ntu/maps/levine'`
- Install dependencies with rosdep:
  ```bash
  source /opt/ros/foxy/setup.bash
  cd ..
  rosdep install -i --from-path src --rosdistro foxy -y
  ```
- Build the workspace: ```colcon build```

# Configuring the simulation
- The configuration file for the simulation is at `f1tenth_gym_ros_ntu/config/sim.yaml`.
- Topic names and namespaces can be configured but is recommended to leave uncahnged.
- The map can be changed via the `map_path` parameter. You'll have to use the full path to the map file in the container. It is assumed that the image file and the `yaml` file for the map are in the same directory with the same name.
- There are 4 maps (levine, levine_closed, Austin, Spielberg). levine_closed is a closed version of levine for min time raceline optimization. Set the map name correctly in sim.yaml

# Running the Simulation

## Terminal 1: Rviz Window
```
cd sim_ws
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

## Terminal 2: Waypoint Marker Script
```
cd sim_ws
source install/setup.bash
ros2 run f1tenth_gym_ros waypoint --ros-args -p map_name:=levine_closed
```
map_name can be anyname from [levine_closed, Austin, Spielberg]. Do turn on the **Marker** in Rviz to visualize the waypoint.

## Terminal 3: Run Pure Pursuit Controller
```
cd sim_ws
source install/setup.bash
ros2 run f1tenth_gym_ros simple_pp
```
The car should now track the waypoint.
