#!/usr/bin/env python3
import sys
import yaml
import rclpy
import threading

sys.path.append("/src/shared/ros_packages/px4_ros_extended/src_py")
from env_wrapper import EnvWrapperNode


class TestWrapperNode:
    def __init__(self, node):
        with open('/src/shared/ros_packages/px4_ros_extended/src_py/params.yaml') as info:
            self.info_dict = yaml.load(info, Loader=yaml.SafeLoader)

        self.env = EnvWrapperNode(node, self.info_dict['obs_shape'], self.info_dict['action_space'],
                                  self.info_dict['max_height'],
                                  self.info_dict['max_side'], self.info_dict['max_vel_z'], self.info_dict['max_vel_xy'],
                                  self.info_dict['eps_pos_xy'])

    def normalize_input(self, inputs_unnormalized):
        inputs_unnormalized[:2] /= self.info_dict['max_side']
        inputs_unnormalized[2] /= self.info_dict['max_height']
        inputs_unnormalized[3:] /= self.info_dict['max_vel_xy']
        return inputs_unnormalized

    def test(self):
        while self.env.reset:  # Waiting for env to stop resetting
            pass
            
        cont_steps = 0
        inputs = self.env.play_env()  # Start landing listening in src_cpp/env.cpp
        while True:
            action = inputs[:3]
            print("Next action: ", action)
            inputs, reward, done = self.env.act(0.3*action, self.normalize_input)
            cont_steps += 1
            if done or (cont_steps % self.info_dict['num-steps']==0 and cont_steps>0):
                self.env.reset_env()
                cont_steps = 0
                while self.env.reset:  # Waiting for env to stop resetting
                    pass
                inputs = self.env.play_env()  # Start landing listening in src_cpp/env.cpp


def spin_thread(node):
    rclpy.spin(node)


if __name__ == '__main__':
    print("Starting Env Wrapper and test node")
    rclpy.init(args=None)
    m_node = rclpy.create_node('agent_node')
    gsNode = TestWrapperNode(m_node)
    x = threading.Thread(target=spin_thread, args=(m_node,))
    x.start()
    gsNode.test()
    x.join()
    rclpy.shutdown()  # Closing Env Wrapper Node
