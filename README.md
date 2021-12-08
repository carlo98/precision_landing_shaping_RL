# Precision landing with physics-informed reinforcement learning
Physics-informed deep reinforcement learning for drone precision landing, docker container for simulation in 
Gazebo-ROS2 dashing with PX4-Autopilot controller. 

Project developed, starting from a problem of [DRAFT PoliTO](https://www.draftpolito.it/), as part of the course 
"AI in Industry" of the University 
of Bologna.

This repository is a work in progress, take a look at the [Project Board](https://github.com/carlo98/precision_landing_physics_informed_RL/projects/1) 
to follow the progress.

# Table of contents
1. [Setup](#setup)
2. [Usage](#usage)
   1. [Train](#train)
   2. [Test](#test)
   3. [Speed-up](#speed)
3. [References](#references)

## Setup <a name="setup"></a>
Modify the absolute paths in "run_docker.sh" to reflect the position of the repository on your computer.

Build and start the docker, it will take same time:
```
cd ..  # Go back to the root of the repository
sudo ./run_docker.sh build
```

Don't worry if this message appears "bash: /src/shared/ros_packages/install/setup.bash: No such file or directory", 
you just need to follow the rest of the setup and the following times it won't happen again.

Once it has finished, in the docker run the following commands, in order to build the packages:
```
cd /src/shared/ros_packages
colcon build --packages-select px4_msgs custom_msgs
colcon build --packages-select px4_ros_com px4_ros_extended
```

## Usage <a name="usage"></a>
To start the docker run:
```
sudo ./run_docker.sh run
```

### Train <a name="train"></a>
Open 2 terminals and run the docker in each one of them, as explained above.

In the first one run:
```
cd /src/shared
./launch_train_ddpg.sh
```

In the second one run:
```
ros2 run px4_ros_extended gazebo_runner.py --train
```

#### Parameters
In "shared/ros_packages/px4_ros_extended/src_py/params.yaml" you can set a few parameters, all of them but 
"train_window_reward" and "test_window_reward", that are used by the jupyter notebook, are used by the agent.

Pay particular attention to the model name, the target dimension and the observation shape, as the test script will need 
to know these values.

#### Plots and Models
In the folder "shared/logs" are saved the rewards in pickle files, you can have a look at the jupyter notebook 
"shared/Log Analysis.ipynb" to use them and print a few plots.

In the folder "shared/models" are saved the actor and critic models, 
the files with "best" in the name contain the best weights found during evaluation; the number at the start of the name 
represents the unique-id of the session.

### Test <a name="test"></a>
Open 2 terminals and run the docker in each one of them, as explained above.

In the first one run:
```
ros2 run px4_ros_extended gazebo_runner.py --test
```

#### Baseline
In the second one run
```
cd /src/shared
./launch_baseline.sh
```

#### Agent
In the second one run
```
cd /src/shared
./launch_test_ddpg.sh N1 <model> N2
```
Where N1 is the run_id, 'model' is the name of the model to be used and N2 is the state shape.

#### Results
The position and velocities for each episode of test are saved in the folder "shared/test_logs".

You can retrieve that information as shown in the jupyter notebook "shared/Log Analysis.ipynb".


### Speed-up & Useful PX4 Parameters <a name="speed"></a>
In order to speed-up the simulation one can start it with these commands, they are already used in the bash script and 
in gazebo_runner.py:
```
PX4_SIM_SPEED_FACTOR=5 HEADLESS=1 make px4_sitl_rtps gazebo
micrortps_agent -t UDP
ros2 run px4_ros_extended ddpg_agent.py -p /use_sim_time:=true
ros2 run px4_ros_extended env -p /use_sim_time:=true
```

In order to avoid following the drone, used in "gazebo_runner.py --test" 
```
PX4_NO_FOLLOW_MODE=1 make px4_sitl_rtps gazebo
```

## References <a name="references"></a>
The initial code for the DDPG algorithm has been taken from [this](https://github.com/vy007vikas/PyTorch-ActorCriticRL) 
github repository.
