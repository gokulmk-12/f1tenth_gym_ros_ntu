# F1TENTH gym environment ROS2 communication bridge
This is a containerized ROS communication bridge for the F1TENTH gym environment that turns it into a simulation in ROS2.

# Videos
Visit this [link](https://drive.google.com/drive/folders/1RMp3XNFU-OyIrURzH7ZhI1Ggcmpawvg-?usp=drive_link) to view some experimental simulation videos.

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
- There are 6 maps (levine, levine_closed, Montreal, Sao Paulo, Austin, Spielberg). levine_closed is a closed version of levine for min time raceline optimization. Set the map name correctly in sim.yaml

# Running the Simulation
## 1) Simple Pure Pursuit Controller

### Terminal 1: Rviz Window
```
cd sim_ws
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

### Terminal 2: Run Pure Pursuit Controller
```
cd sim_ws
source install/setup.bash
ros2 run f1tenth_gym_ros simple_pp
```
Do change the variable `map_name` to exact map name as defined in `sim.yaml`. The car should now track the waypoint.

## 2) Residual RL (SAC) Controller - No Obstacles - levine_closed map

Ensure that the `map_name` and `map_path` are correct and match with the csv file name in tracks folder

### Terminal 1: Rviz Window
```
cd sim_ws
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
The script to run RL is test.py in RL folder. Make sure the variables `exp_no`, `algo`, `logs_type`, `obstacle_type` match with the trained weights in `logs/` folder.

### Terminal 2: Run Pure Pursuit Controller
```
cd sim_ws/src/f1tenth_gym_ros
python3 RL/test.py
```
If everything goes correct, you should see the car moving with the residual RL controller.

## 3) Residual RL (SAC) Controller - Static Obstacles - levine_closed_static_obstacle map

Do the following changes in `sim.yaml` 
```
# map parameters
map_name: 'levine_closed_static_obstacle'
map_path: '/sim_ws/src/f1tenth_gym_ros/maps/levine_closed_static_obstacle'
```
Edit the `obstacle_type` variable value to `static_obstacles`, while keeping others fixed. Build the workspace and do the same steps as in explained in Part 2.

## 4) Residual RL (SAC) Controller - Dynamic Obstacles - levine_closed map

Do the following changes in `sim.yaml` 
```
# map parameters
map_name: 'levine_closed'
map_path: '/sim_ws/src/f1tenth_gym_ros/maps/levine_closed'

# opponent parameters
num_agent: 2
```

Do the following changes in `RL/test.py`
```
env = F110Gym(is_train=False, is_opp=True)
obs = env.reset()

exp_no = 1
algo = "SAC"
logs_type = "obs_logs"
obstacle_type="dynamic_obstacles_2"
```
Ensure that the `is_opp` variable is set to True. Build the workspace and do the same steps as in explained in Part 2. Now you can see two racecars spawn. One runs the Simple Pure Pursuit and the other runs the Residual RL controller.
