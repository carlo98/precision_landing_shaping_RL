import numpy as np
import torch
import shutil


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :param tau
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param episode_count:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self, n_steps):
        dx = self.theta * (self.mu - self.X)
        
        # Coefficient for number of steps chosen in order to have noise in range -0.3 and 0.3 after 1000k steps ~ 150 steps/episode & 6k episodes
        dx = dx + self.sigma * 0.9999985**n_steps * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


# use this to plot Ornstein Uhlenbeck random motion
if __name__ == '__main__':
    ou = OrnsteinUhlenbeckActionNoise(1)
    states = []
    for i in range(1000000, 1100000):
        states.append(ou.sample(i))
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
