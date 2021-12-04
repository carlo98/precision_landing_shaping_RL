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
   1. [PX4-Autopilot](#px4)
   2. [Docker Build](#docker)
   3. [Jupyter Notebook](#notebook)
2. [Usage](#usage)
   1. [Train](#train)
   2. [Test](#test)
   3. [Speed-up](#speed)
3. [References](#references)

## Setup <a name="setup"></a>

### PX4 Autopilot <a name="px4"></a>
Clone [PX4-AutoPilot](https://github.com/PX4/PX4-Autopilot) in the "shared" folder.
```
cd shared
git clone https://github.com/PX4/PX4-Autopilot.git
```

### Docker Build <a name="docker"></a>
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

Start PX4-Autopilot in order to perform the build, it will take same time
```
cd /src/shared/PX4-Autopilot
make px4_sitl_rtps gazebo
```

### Jupyter Notebook <a name="notebook"></a>
In order to use the "Log Analysis.ipynb" to look at the results and rewards you will have to install a few things in 
your computer, 
outside of the docker run:
```
sudo apt install jupyter-notebook  # If this doesn't work search for jupyter notebook installation in internet, there's plenty of resources
sudo apt install pandas
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

#### Plots and Models
In the folder "shared/logs" are saved the rewards in pickle files, you can have a look at the jupyter notebook 
"Log Analysis.ipynb" to use them and print a few plots.

In the folder "shared/models" are saved the actor and critic models, 
the files with "best" in the name contain the best weights found during evaluation; the number at the start of the name 
represents the unique-id of the session.

### Test <a name="test"></a>
Open 2 terminals and run the docker in each one of them, as explained above.

To start Gazebo:
```
ros2 run px4_ros_extended gazebo_runner.py --test
```

#### Baseline
Once gazebo is started in a new terminal run
```
cd /src/shared
./launch_baseline.sh
```

#### Agent
Once gazebo is started in a new terminal run
```
cd /src/shared
./launch_test_ddpg.sh
```


### Speed-up <a name="speed"></a>
In order to speed-up the simulation one can start it with these commands, they are already used in the bash script and 
in gazebo_runner.py:
```
PX4_SIM_SPEED_FACTOR=2 HEADLESS=1 make px4_sitl_rtps gazebo
micrortps_agent -t UDP
ros2 run px4_ros_extended ddpg_agent.py -p /use_sim_time:=true
ros2 run px4_ros_extended env -p /use_sim_time:=true
```

## References <a name="references"></a>
The initial code for the DDPG algorithm has been taken from [this](https://github.com/vy007vikas/PyTorch-ActorCriticRL) 
github repository.
