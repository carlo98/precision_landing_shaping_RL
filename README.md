# Precision landing with physics-informed reinforcement learning
Deep NN trained with physics-informed reinforcement learning for drone precision landing.

# Table of contents
1. [Setup](#setup)
2. [Usage](#usage)
    1. [Train](#train)
    2. [Test](#test)
3. [References](#references)

## Setup <a name="setup"></a>
Clone [PX4-AutoPilot](https://github.com/PX4/PX4-Autopilot) in the "shared" folder.
```
cd shared
git clone https://github.com/PX4/PX4-Autopilot.git
```

Modify the absolute paths in "run_docker.sh" to reflect the position of the repository on your computer.

Build and start the docker, it will take same time:
```
sudo ./run_docker.sh build
```

Don't worry if this message appears "bash: /src/shared/ros_packages/install/setup.bash: No such file or directory", you just need to follow the rest of the setup and the following times it won't happen again.

Once it has finished, in the docker run the following commands, in order to build the packages:
```
colcon build --packages-select px4_msgs custom_msgs
colcon build --packages-select px4_ros_com px4_ros_extended
```

## Usage <a name="usage"></a>
To start the docker run:
```
sudo ./run_docker.sh run
```

To start gazebo:
```
cd /src/shared/PX4-Autopilot/
make px4_sitl_rtps gazebo
```

### Train <a name="train"></a>

### Test <a name="test"></a>
Open a terminal and divide it with "tmux" in 5 command lines or open 5 terminals and run the docker in each one of them, as explained above.

Once gazebo is started, in a new terminal run 
```
micrortps_agent -t UDP
```

#### Baseline
Then takeoff with:
```
ros2 run px4_ros_extended takeoff
```

Once the drone is stable run in two different windows:
```
ros2 run px4_ros_extended baseline_prec_land
ros2 run px4_ros_extended land
```

Once the first messages appear in the window in which you have runned the land node, stop the takeoff node with "ctrl-c."

#### DDPG

## References <a name="references"></a>
The code for the PPO algoritm and the memory has been taken from [this](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) github repository.

The code for the DDPG algorithm has been taken from [this](https://github.com/vy007vikas/PyTorch-ActorCriticRL) github repository.
