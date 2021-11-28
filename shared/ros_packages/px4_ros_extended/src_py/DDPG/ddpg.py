from __future__ import division
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import DDPG.utils as utils
from DDPG.model_paper import Critic, Actor


class DDPG:
    def __init__(self, state_dim, action_dim, ram, lr=0.001, gamma=0.99, tau=0.001, batch_size=128, epochs=3):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param ram: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ram = ram
        self.iter = 0
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim, self.action_dim).float()
        self.target_actor = Actor(self.state_dim, self.action_dim).float()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

        self.critic = Critic(self.state_dim, self.action_dim).float()
        self.target_critic = Critic(self.state_dim, self.action_dim).float()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.path_models = "/src/shared/models"
        if not os.path.isdir(self.path_models):
            os.mkdir(self.path_models)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).float())
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).float())
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + self.noise.sample()
        return new_action

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        for j in range(self.epochs):
            tot_loss_critic = 0.0
            tot_loss_actor = 0.0
            s1_tot, a1_tot, r1_tot, s2_tot = self.ram.sample(len(self.ram.buffer))
            for i in range(int(len(s1_tot)/self.batch_size)):
                s1 = np.array(s1_tot)[i*self.batch_size:i*self.batch_size+self.batch_size]
                a1 = np.array(a1_tot)[i*self.batch_size:i*self.batch_size+self.batch_size]
                r1 = np.array(r1_tot)[i*self.batch_size:i*self.batch_size+self.batch_size]
                s2 = np.array(s2_tot)[i*self.batch_size:i*self.batch_size+self.batch_size]

                s1 = Variable(torch.Tensor(s1))
                a1 = Variable(torch.Tensor(a1))
                r1 = Variable(torch.Tensor(r1))
                s2 = Variable(torch.Tensor(s2))

                # ---------------------- optimize critic ----------------------
                # Use target actor exploitation policy here for loss evaluation
                a2 = self.target_actor.forward(s2).detach()
                next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
                # y_exp = r + gamma*Q'( s2, pi'(s2))
                y_expected = r1 + self.gamma * next_val
                # y_pred = Q( s1, a1)
                y_predicted = torch.squeeze(self.critic.forward(s1, a1))
                # compute critic loss, and update the critic
                loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()

                # ---------------------- optimize actor ----------------------
                pred_a1 = self.actor.forward(s1)
                loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()

                tot_loss_critic += loss_critic.item()
                tot_loss_actor += loss_actor.item()

            print("Epoch: " + str(j) + " Loss Critic: " + str(tot_loss_critic/self.epochs) + " Loss Actor: "
                  + str(tot_loss_actor/self.epochs))

        utils.soft_update(self.target_actor, self.actor, self.tau)
        utils.soft_update(self.target_critic, self.critic, self.tau)

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), self.path_models + '/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), self.path_models + '/' + str(episode_count) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(self.path_models + '/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(self.path_models + '/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded successfully')
