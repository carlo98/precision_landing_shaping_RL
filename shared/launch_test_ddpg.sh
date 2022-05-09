#!/bin/bash

SESSION=$USER

# Create new session (-s session name, -d -> not attach to the new session)
tmux new-session -d -s=$SESSION

# Create one window (-t -> target session, -n -> window name)
tmux new-window -t $SESSION:2 -n 'micrortps'
tmux send-keys "micrortps_agent -t UDP" C-m

sleep 2

tmux new-window -t $SESSION:1 -n 'env + agent'
tmux send-keys "ros2 run px4_ros_extended test_ddpg_agent.py $1 $2 $3 $4 $5" C-m

tmux split-window -h -t $SESSION:1
tmux send-keys "ros2 launch px4_ros_extended env.launch.py" C-m

tmux attach-session -t $SESSION:1
