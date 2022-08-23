import scipy.stats as stats
import matplotlib.pyplot as plt
from motion_equations import Quadcopter6DOF
import numpy as np

# fix random state
SIMULATION_RUNS = 25
SAMPLES_BUFFER = 50 # This value should be adjusted according to the simulation length

#create a distribution of initial condition parameters
pitch_cmd_normal = stats.norm(loc=0, scale=10)
roll_cmd_normal = stats.norm(loc=0, scale=0.3)
climb_cmd_normal = stats.norm(loc=-50, scale=100) # the drone moves mainly torward the ground.
yaw_cmd_normal = stats.norm(loc=0, scale=0.3) # does not influence the dynamic of interest
# create distribution of command time 
time_cmd_gamma = stats.gamma(a=2)

# initialize quadcopter object
quad = Quadcopter6DOF()

# Set parameters for the integration
T_START=0
T_END=20
# calculate number of integration steps required
INTEGRATION_STEP = 1 / 50
 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# run Monte Carlo Simulation for with those distribution of parameters above
for i in range(SIMULATION_RUNS):

    # sample the distributions to create a buffer
    pitch_samples = pitch_cmd_normal.rvs(size=SAMPLES_BUFFER)
    roll_samples = roll_cmd_normal.rvs(size=SAMPLES_BUFFER)
    climb_samples = climb_cmd_normal.rvs(size=SAMPLES_BUFFER)
    yaw_samples = yaw_cmd_normal.rvs(size=SAMPLES_BUFFER) 
    time_cmd_samples = time_cmd_gamma.rvs(size=SAMPLES_BUFFER)

    # create command array (2D)
    control_sequence = quad.control_sequence(
        t_start=T_START,
        t_end=T_END, 
        step=INTEGRATION_STEP, 
        time_cmd_samples=time_cmd_samples, 
        pitch_sample=pitch_samples, 
        roll_samples=roll_samples, 
        climb_samples=climb_samples, 
        yaw_samples=yaw_samples
        )
    

    # solver setting and run
    solver_t = 0 # inialize solver time
    # get initial conditions
    x_state = quad.initial_conditions(xE=0, yE=0, hE=2)

    while np.around(solver_t, 3) < T_END:
        u_control = quad.control_inputs(solver_t, control_sequence=control_sequence, trim_rpm=3265)
        x_integration = x_state[-1,:]
        x_state_update = quad.solver_rk4(x_state=x_integration, u=u_control, step=INTEGRATION_STEP)

        solver_t = solver_t+INTEGRATION_STEP
        x_state = np.vstack((x_state, x_state_update))
        
        # stop simulation for negative heights
        if quad.hit_ground(x_state_update, step=INTEGRATION_STEP) == True:
            break

    ax.plot(x_state[:,9], x_state[:,10], x_state[:,11])
    #plt.plot(x_state[:,9],x_state[:,11],'bo-', label=f"MC {i}")
plt.show()