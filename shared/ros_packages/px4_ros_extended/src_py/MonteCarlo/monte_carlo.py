"""
This class performs a Monte Carlo Search at different heights starting from a given threshold.
"""
import scipy.stats as stats
import pickle
import sys
import numpy as np

sys.path.append("/src/shared/ros_packages/px4_ros_extended/src_py")

# from ode_models.projectile.motion_equations import ProjectileMotion
from ode_models.quadcopter.motion_equations import Quadcopter6DOF


class MonteCarlo:
    def __init__(self, id_file, path_logs, action_dim, start_height=100, simulation_runs=500, num_samples=100,
                 scale=0.05):
        """
        Initialize MonteCarlo object
        :param id_file: Used to log actions, first part of each file name
        :param path_logs: Folder in which to log actions
        :param action_dim: action's dimension
        :param start_height: Height at which to start Monte Carlo Simulation (centimeters)
        :param simulation_runs: number of steps for Monte Carlo Simulation
        :param num_samples: Number of samples for each step, i.e. for start_height/steps_dist times
        :param scale: standard deviation for gaussian distribution
        :return:
        """
        self.id_file = id_file
        self.path_logs = path_logs
        self.simulation_runs = simulation_runs
        self.start_height = start_height
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.scale = scale
        self.random_state = 42
        self.episodes = 0

        self.action_log = []

    def generate_samples(self, model_state):
        # create a distribution of initial condition parameters
        pitch_cmd_normal = stats.norm(loc=0, scale=10)
        roll_cmd_normal = stats.norm(loc=0, scale=0.3)
        # loc negative, the drone moves mainly toward the ground.
        climb_cmd_normal = stats.norm(loc=model_state[2], scale=100)
        yaw_cmd_normal = stats.norm(loc=0, scale=0.3)  # does not influence the dynamic of interest
        # create distribution of command time
        time_cmd_gamma = stats.gamma(a=2)

        # initialize quadcopter object
        quad = Quadcopter6DOF()

        # Set parameters for the integration
        t_start = 0
        t_end = 20
        # calculate number of integration steps required
        integration_step = 1 / 50

        # run Monte Carlo Simulation for with those distribution of parameters above
        end_poses = []
        for i in range(self.simulation_runs):
            # sample the distributions to create a buffer
            pitch_samples = pitch_cmd_normal.rvs(size=self.num_samples)
            roll_samples = roll_cmd_normal.rvs(size=self.num_samples)
            climb_samples = climb_cmd_normal.rvs(size=self.num_samples)
            yaw_samples = yaw_cmd_normal.rvs(size=self.num_samples)
            time_cmd_samples = time_cmd_gamma.rvs(size=self.num_samples)

            # create command array (2D)
            control_sequence = quad.control_sequence(
                t_start=t_start,
                t_end=t_end,
                step=integration_step,
                time_cmd_samples=time_cmd_samples,
                pitch_sample=pitch_samples,
                roll_samples=roll_samples,
                climb_samples=climb_samples,
                yaw_samples=yaw_samples
            )

            # solver setting and run
            solver_t = 0  # initialize solver time
            # get initial conditions
            x_state = quad.initial_conditions(xE=0, yE=0, hE=2)
            end_poses.append(x_state[-1, 9:12])

            while np.around(solver_t, 3) < t_end:
                u_control = quad.control_inputs(solver_t, control_sequence=control_sequence, trim_rpm=3265)
                x_integration = x_state[-1, :]
                x_state_update = quad.solver_rk4(x_state=x_integration, u=u_control, step=integration_step)

                solver_t = solver_t + integration_step
                x_state = np.vstack((x_state, x_state_update))

                # stop simulation for negative heights
                if quad.hit_ground(x_state_update, step=integration_step):
                    break

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
