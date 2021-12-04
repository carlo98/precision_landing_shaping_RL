#!/bin/bash

SESSION=$USER

# Create new session (-s session name, -d -> not attach to the new session)
tmux new-session -d -s=$SESSION

# Create one window (-t -> target session, -n -> window name)
tmux new-window -t $SESSION:2 -n 'micrortps'
tmux send-keys "micrortps_agent -t UDP" C-m

sleep 2

tmux new-window -t $SESSION:1 -n 'env + baseline'
tmux send-keys "ros2 run px4_ros_extended baseline_prec_land" C-m

tmux split-window -h -t $SESSION:1
tmux send-keys "ros2 run px4_ros_extended env" C-m

tmux attach-session -t $SESSION:1