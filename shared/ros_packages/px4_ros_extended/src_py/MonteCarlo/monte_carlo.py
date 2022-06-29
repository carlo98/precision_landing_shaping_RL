"""
This class performs a Monte Carlo Search at different heights starting from a given threshold.
"""
import scipy.stats as stats
import pickle
import sys

sys.path.append("/src/shared/ros_packages/px4_ros_extended/src_py")

from ode_models.projectile.motion_equations import ProjectileMotion


class MonteCarlo:
    def __init__(self, id_file, path_logs, action_dim, start_height=100, simulation_steps=500, num_samples=100,
                 scale=0.05):
        """
        Initialize MonteCarlo object
        :param id_file: Used to log actions, first part of each file name
        :param path_logs: Folder in which to log actions
        :param action_dim: action's dimension
        :param start_height: Height at which to start Monte Carlo Simulation (centimeters)
        :param simulation_steps: number of steps for Monte Carlo Simulation
        :param num_samples: Number of samples for each step, i.e. for start_height/steps_dist times
        :param scale: standard deviation for gaussian distribution
        :return:
        """
        self.id_file = id_file
        self.path_logs = path_logs
        self.simulation_steps = simulation_steps
        self.start_height = start_height
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.scale = scale
        self.random_state = 42
        self.episodes = 0

        self.action_log = []

    def generate_samples(self, model_state):
        # create a distribution of initial condition parameters
        c_normal = stats.norm(loc=model_state[2], scale=self.scale)
        v0_normal = stats.norm(loc=model_state[5], scale=2)
        phi0_normal = stats.norm(loc=45, scale=3)

        # sample the distribution according to the number of simulation steps required
        c_samples = c_normal.rvs(size=self.simulation_steps, random_state=self.random_state)
        v0_samples = v0_normal.rvs(size=self.simulation_steps, random_state=self.random_state)
        phi0_samples = phi0_normal.rvs(size=self.simulation_steps, random_state=self.random_state)
        end_poses = []
        for i in range(self.simulation_steps):
            projectile = ProjectileMotion(mass=1, radius=0.05, c=c_samples[i])
            u0 = projectile.initial_conditions(z0=0, v0=v0_samples[i], phi0=phi0_samples[i])
            x, z = projectile.solve_motion(u0, t_end=30, steps_integration=1000)
            end_poses.append([x[-1], x[-1], z[-1]])  # TODO: add y and velocities

        self.action_log.append(end_poses)
        return end_poses
        
    def get_start_height(self):
        """
        Returns start height in metres
        """
        return self.start_height / 100

    def reset(self):
        self.episodes += 1
        self._log()

    def _log(self):
        filename = '/log_' + str(self.id_file) + "_" + str(self.episodes) + ".pkl"
        with open(self.path_logs+filename, "wb") as pkl_f:
            pickle.dump(self.action_log, pkl_f)
