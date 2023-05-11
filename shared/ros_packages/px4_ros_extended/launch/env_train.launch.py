import os
import yaml

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    with open('/src/shared/ros_packages/px4_ros_extended/src_py/params.yaml') as info:
        info_dict = yaml.load(info, Loader=yaml.SafeLoader)
    
    param = []
    for key in info_dict.keys():
        param.append({key: info_dict[key]})

    flight_plan_node = Node(
        package = 'px4_ros_extended', 
        executable = 'env',
        name = 'env_node',
        output = {
            'stdout': 'screen',
            'stderr': 'screen',
        },
        parameters = param,
        arguments = ['__log_level:=info', "-p /use_sim_time:=true"],
    )


    #! Node execution
    ld = LaunchDescription()

    ld.add_action(flight_plan_node)

    return ld
