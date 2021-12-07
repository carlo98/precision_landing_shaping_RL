#!/bin/bash

SESSION=$USER

# Create new session (-s session name, -d -> not attach to the new session)
tmux new-session -d -s=$SESSION

# Create one window (-t -> target session, -n -> window name)
tmux new-window -t $SESSION:1 -n 'env + agent'
tmux send-keys "ros2 run px4_ros_extended ddpg_agent.py -p /use_sim_time:=true" C-m

sleep 3  # Waiting for micrortps_agent process started by agent

tmux split-window -h -t $SESSION:1
tmux send-keys "ros2 run px4_ros_extended env -p /use_sim_time:=true" C-m

tmux attach-session -t $SESSION:1
