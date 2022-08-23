"""
Motion equation of a quadricopter
"""

import re
import numpy as np
from scipy.integrate import solve_ivp, odeint, ode
from scipy.optimize import fsolve

class Quadcopter6DOF(object):

    def __init__(self, mass=0.1, ixx=0.00062, iyy=0.00113, dx=0.114, dy=0.0825, dz=0, g=9.81, rho=1.225, rtd=57.3) -> None:
        # Physical Constants
        self.mass = mass  # kg
        self.ixx = ixx  # kg-m^2
        self.iyy = iyy  # kg-m^2
        self.izz = 0.9 * (self.ixx + self.iyy)  # kg-m^2 (Assume nearly flat object, z=0)
        self.dx = dx  # m
        self.dy = dy  # m
        self.dz = dz  # m
        self.g = g  # m/s/s
        self.rho = rho #kg/m^3  at MSL
        self.rtd = rtd
        self.dtr = 1 / self.rtd

    def propeller_design(self) -> dict:
        radius = 0.0762   # propeller length/ disk radius (m)
        disk_area = np.pi * radius ** 2
        a = 5.7  # lift curve slope used in example in Stevens & Lewis
        b = 2  # number of blades
        c = 0.0274  # mean chord length (m)
        eta = 1  # propeller efficiency

        # Manufacturer propeller length x pitch specification:
        p_diameter = 6  #inches
        p_pitch = 3   #inches

        theta0 = 2*np.arctan2(p_pitch, (2 * np.pi * 3/4 * p_diameter/2))
        theta1 = -4 / 3 * np.arctan2(p_pitch, 2 * np.pi * 3/4 * p_diameter/2)

        return {"radius": radius,
                "disk_area": disk_area,
                "rho":self.rho,
                "a": a,
                "b": b,
                "c": c,
                "eta": eta,
                "theta0": theta0,
                "theta1": theta1}

    def thrust_equation(self, vi, prop_params):
        """
        Propeller Thrust equations as a function of propeller induced velocity, vi

        :param vi:
        :param propeller_parameters:
        :return:
        """
        #prop_params = (R, A, rho, a, b, c, eta, theta0, theta1, U, V, W, omega)


        # Calculate local airflow velocity at propeller with vi, V'
        v_prime = np.sqrt(prop_params['U'] ** 2 + prop_params['V'] ** 2 + (prop_params['W'] - vi) ** 2)

        # Calculate Thrust averaged over one revolution of propeller using vi
        thrust = 1 / 4 * prop_params['rho'] * \
                 prop_params['a'] * \
                 prop_params['b'] * \
                 prop_params['c'] * \
                 prop_params['radius'] * \
                 ((prop_params['W'] - vi) * prop_params['omega'] * prop_params['radius'] + 2 / 3 *
                  (prop_params['omega'] * prop_params['radius']) ** 2 * (prop_params['theta0'] +
                                                                         3 / 4 * prop_params['theta1']) +
                  (prop_params['U'] ** 2 + prop_params['V'] ** 2) * (prop_params['theta0'] + 1 / 2 *
                                                                     prop_params['theta1']))

        # Calculate residual for equation: Thrust = mass flow rate * delta Velocity
        residual = prop_params['eta'] * 2 * vi * prop_params['rho'] * prop_params['disk_area'] * v_prime - thrust

        return residual

    def thrust(self, x, u, dx, dy):
        """
        Inputs: Current state x[k], Commanded Propeller RPM inputs u[k],
        Propeller location distances dx, dy (m)
        Returns: Thrust vector for 4 propellers (Newtons)

        :param x:
        :param u:
        :param dx:
        :param dy:
        :return:
        """
        # Inputs: Current state x[k], Commanded Propeller RPM inputs u[k],
        #         Propeller location distances dx, dy (m)
        # Returns: Thrust vector for 4 propellers (Newtons)

        # Propeller Configuration parameters
        prop_design = self.propeller_design()

        # Local velocity at propeller from vehicle state information
        ub, vb, wb = x[0], x[1], x[2]
        p, q, r = x[3], x[4], x[5]

        # Transform velocity to local propeller location:
        #     [U,V,W] = [ub,vb,wb] + [p,q,r] x [dx,dy,0]

        U = ub - r * dy
        V = vb + r * dx
        W = wb - q * dx + p * dy

        # Convert commanded RPM to rad/s
        omega = 2 * np.pi / 60 * u

        # add create velocity dictionary
        prop_velocity = {"U": U,
                         "V": V,
                         "W": W,
                         "omega": omega}
        # Collect propeller config, state, and input parameters
        prop_params = {**prop_design, **prop_velocity}
        #prop_params = (R, A, rho, a, b, c, eta, theta0, theta1, U, V, W, omega)

        # Numerically solve for propeller induced velocity, vi
        # using nonlinear root finder, fsolve, and prop_params
        vi0 = 0.01  # initial guess for vi
        vi = fsolve(self.thrust_equation, vi0, args=prop_params)

        # Plug vi back into Thrust equation to solve for T
        vprime = np.sqrt(U ** 2 + V ** 2 + (W - vi) ** 2)
        thrust = prop_params['eta']  * 2 * vi * prop_params['rho'] * prop_params['disk_area'] * vprime
        
        return thrust

    def torque(self, F, dx, dy):
        """
        Torque estimation - PLACEHOLDER
        Returns torque about cg given thrust force and dx,dy distance from cg

        :param F:
        :param dx:
        :param dy:
        :return:
        """
        return 0

    def initial_conditions(self, u=0, v=0, w=0, p=0, q=0, r=0, phi=0, the=0, psi=0, xE=0, yE=0, hE=2):
        """
        x = [u, v, w, p, q, r, phi, the, psi, xE, yE, hE]
        """
        return np.array([[u, v, w, p, q, r, phi, the, psi, xE, yE, hE]])

    def state_derivate(self, x, u):
        """
        Nonlinear Dynamics Equations of Motion
        Inputs: state vector (x), input vector (u)
        Returns: time derivative of state vector (xdot)
        
        State Vector Reference:
        idx  0, 1, 2, 3, 4, 5,  6,   7,   8,   9, 10, 11
        x = [u, v, w, p, q, r, phi, the, psi, xE, yE, hE]
        """
        
        # Store state variables in a readable format
        ub, vb, wb = x[0], x[1], x[2]
        p, q, r = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7],  x[8]
        xE, yE, hE = x[9], x[10], x[11]
        
        # Calculate forces from propeller inputs (u)
        F1 = self.thrust(x, u[0],  self.dx,  self.dy)
        F2 = self.thrust(x, u[1], -self.dx, -self.dy)
        F3 = self.thrust(x, u[2],  self.dx, -self.dy)
        F4 = self.thrust(x, u[3], -self.dx,  self.dy)
        Fz = F1 + F2 + F3 + F4
        L = (F1 + F4) * self.dy - (F2 + F3) * self.dy
        M = (F1 + F3) * self.dx - (F2 + F4) * self.dx
        N = -self.torque(F1, self.dx, self.dy) - self.torque(F2, self.dx, self.dy) + \
            self.torque(F3, self.dx, self.dy) + self.torque(F4, self.dx, self.dy)
        
        # Pre-calculate trig values
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)  
        spsi = np.sin(psi)
        
        # Calculate the derivative of the state matrix using EOM
        xdot = np.zeros(12)
        
        xdot[0] = -self.g * sthe + r * vb - q * wb  # = udot
        xdot[1] = self.g * sphi * cthe - r * ub + p * wb # = vdot
        xdot[2] = 1 / self.mass * (-Fz) + self.g * cphi * cthe + q * ub - p * vb # = wdot
        xdot[3] = 1 / self.ixx * (L + (self.iyy - self.izz) * q * r)  # = pdot
        xdot[4] = 1 / self.iyy * (M + (self.izz - self.ixx) * p * r)  # = qdot
        xdot[5] = 1 / self.izz * (N + (self.ixx - self.iyy) * p * q)  # = rdot
        xdot[6] = p + (q * sphi + r * cphi) * sthe / cthe  # = phidot
        xdot[7] = q * cphi - r * sphi  # = thetadot
        xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
        xdot[9] = cthe * cpsi * ub + (- cphi * spsi + sphi * sthe * cpsi) * vb + \
            ( sphi * spsi + cphi * sthe * cpsi) * wb  # = xEdot   
        xdot[10] = cthe * spsi * ub + (cphi * cpsi + sphi * sthe * spsi) * vb + \
            (- sphi * cpsi + cphi * sthe * spsi) * wb # = yEdot
        xdot[11] = -1 * (- sthe * ub + sphi * cthe * vb + cphi * cthe * wb) # = hEdot
        
        return xdot
    
    def control_sequence(self, t_start, t_end, step, time_cmd_samples, pitch_sample, roll_samples, climb_samples, yaw_samples):
        """
        Create control seauence with 
        dt, pitch, roll, climb, raw
        The function is designed to overestimate the T_END if the simulation

        Args:
            t_start (_type_): _description_
            t_end (_type_): _description_
            step (_type_): _description_
            time_cmd_samples (_type_): _description_
            pitch_sample (_type_): _description_
            roll_samples (_type_): _description_
            climb_samples (_type_): _description_
            yaw_samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_index_time = np.argmin(time_cmd_samples.cumsum() <= t_end)

        _time = t_start
        control_sequence = np.zeros((1, 5))
        for idx, repeat in enumerate (time_cmd_samples[:max_index_time+1]):
            
            # TODO complete the control sequence - sample cmd points to get
            # to the complete sequence
            repeat_var = np.around(repeat/step, 0)

            delta_t = np.around(np.repeat(step, repeat_var), 3)
            delta_t[0] += _time
            delta_t = delta_t.cumsum()

            u_pitch = np.repeat(pitch_sample[idx], repeat_var)
            u_roll = np.repeat(roll_samples[idx], repeat_var)
            u_climb = np.repeat(climb_samples[idx], repeat_var)
            u_yaw = np.repeat(yaw_samples[idx], repeat_var)
            _time = delta_t[-1]

            # stack cmd list into a 2D array
            control_tmp =  np.stack((delta_t, u_pitch, u_roll, u_climb, u_yaw), axis=-1)
            # append to control sequence
            control_sequence = np.vstack((control_sequence, control_tmp))
        
        return np.around(control_sequence, 5)


    def control_inputs(self, t, control_sequence, trim_rpm=3200):
        """
        Inputs: Current state x[k], time t
        Trim RPM for all 4 propellers to provide thrust for a level hover
        Returns: Control inputs u[k]        
        """ 
        t = np.around(t, 5) # TODO remove this walk-around

        u_control = np.squeeze(control_sequence[control_sequence[:, 0] == t])
        # unpack controls
        u_pitch = u_control[1]
        u_roll = u_control[2]
        u_climb = u_control[3]
        u_yaw = u_control[4]

        # RPM command based on pitch, roll, climb, yaw commands
        u1 = trim_rpm + ( u_pitch + u_roll + u_climb - u_yaw) / 4
        u2 = trim_rpm + ( - u_pitch - u_roll + u_climb - u_yaw) / 4
        u3 = trim_rpm + ( u_pitch - u_roll + u_climb + u_yaw) / 4
        u4 = trim_rpm + ( - u_pitch + u_roll + u_climb + u_yaw) / 4

        # concatenate commands horizzomtaly 
        u = np.stack((u1, u2, u3, u4), axis=-1)

        return u

    def hit_ground(self, x, step):
        # The drone is on ground if the z-coordinate is 0.
        if np.abs(x[11] <= 0.5*step):
            hit_check = True
        else:
            hit_check = False
        return  hit_check

    def solve_motion(self, x, t_start=0, t_end=60, steps_integration=1200, verbose=False):
        """This method is a wrapper aroung the solve_ivp function of scipy. It is used to 
        solve the ODEs of motion.

        Args:
            x (tuple): Initial condition tuple
            t_start (int, optional): Initial time of integration. Defaults to 0.
            t_end (int, optional): Final time of integration. This parameter is also the max time of integration. 
            Defaults to 60.
            steps_integration (int, optional): Number of steps that will be used to split the integration time. 
            The higher this value is the better is the approximation of the solution. Defaults to 1000.

        Returns:
            array: Solution of the ODEs.
        Reference: https://github.com/bobzwik/Quad_Exploration/tree/master/Simulation
        """

        # event termination
        event_hit_ground = self.hit_ground
        event_hit_ground.__dict__['terminal'] = True
        event_hit_ground.__dict__['direction'] = -1

        # initilize the solver
        solver_ode = solve_ivp(
            self.state_derivate,
            (t_start, t_end), 
            x, 
            dense_output=True, 
            events=(event_hit_ground),
            args=(t)
            )
        
        # print some information about the calculation
        if verbose:
            print('Time to ground = {:.2f} s'.format(solver_ode.t_events[0][0]))

        # define grid of time points from 0 until impact time.
        t = np.linspace(0, solver_ode.t_events[0][0], steps_integration)
        # retrieve the solution for the time grid and plot the trajectory.
        results = solver_ode.sol(t)

        return results

    def solver_dopri5(self, x, t_end=60, steps_integration=1200):

        integrator = ode(self.state_derivate).set_integrator('dopri5', first_step='0.025', atol='10e-4', rtol='10e-4')
        integrator.set_initial_value(x, 0)
        t_step = t_end/steps_integration

        return integrator, t_step

    def solver_rk4(self, x_state, u, step):
            # Inputs: x[k], u[k], dt (time step, seconds)
            # # Returns: x[k+1]

            # TODO implementare controllo della quota del drone

            # Calculate slope estimates
            K1 = self.state_derivate(x_state, u)
            K2 = self.state_derivate(x_state + K1 * step / 2, u)
            K3 = self.state_derivate(x_state + K2 * step / 2, u)
            K4 = self.state_derivate(x_state + K3 * step, u)
            
            # Calculate x[k+1] estimate using combination of slope estimates
            x_state_next = x_state + 1/6 * (K1 + 2*K2 + 2*K3 + K4) * step
            
            return x_state_next