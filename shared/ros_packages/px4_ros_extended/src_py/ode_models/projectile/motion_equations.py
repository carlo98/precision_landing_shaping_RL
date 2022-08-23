"""
Motion equation of a ballistic projectile

Resources for theoretical and code reference:
https://scipython.com/book2/chapter-8-scipy/examples/a-projectile-with-air-resistance/
https://farside.ph.utexas.edu/teaching/336k/Newton/node29.html"

Integrations methods:
https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html

"""
import numpy as np
from scipy.integrate import solve_ivp, odeint 


class ProjectileMotion(object):

    def __init__(self, mass, radius, c, rho=1.28, gravity=9.81) -> None:
        self.mass = mass #mass (kg).
        self.radius = radius #projectile radius (m)  
        self.rho = rho #density (kg.m-3) 
        self.g =  gravity #acceleration (m.s-2)
        self.area = np.pi * self.radius**2 #area (m2)
        self.k = 0.5 * c * self.rho * self.area # resistence coefficient

    def initial_conditions(self, z0, v0, phi0):
        """Method included to calculate and return the initial condition of the 
        problem at hand.

        Args:
            u0 (float): Initial height of the launch (m)
            v0 (float): Initial speed of the launch (m/s).
            phi0 (float): Initial launch angle with respect to the horizontal (deg).
            
        Returns:
            tuple: tuple containing the initial condition.
        """
        phi0 = np.radians(phi0)
        u0 = 0, v0 * np.cos(phi0), z0, v0 * np.sin(phi0) # initial condition tuple
        return u0

    def equation_derivate(self, t, u0):
        x, xdot, z, zdot = u0
        speed = np.hypot(xdot, zdot)
        xdotdot = -self.k/self.mass * speed * xdot
        zdotdot = -self.k/self.mass * speed * zdot - self.g
        return xdot, xdotdot, zdot, zdotdot

    def hit_target(self, t, u):
        # We've hit the target if the z-coordinate is 0.
        return u[2]

    def max_height(self, t, u):
        # The maximum height is obtained when the z-velocity is zero.
        return u[3]

    def solve_motion(self, u0, t_start=0, t_end=60, steps_integration=1000, verbose=False):
        """This method is a wrapper aroung the solve_ivp function of scipy. It is used to 
        solve the ODEs of motion.

        Args:
            u0 (tuple): Initial condition tuple
            t_start (int, optional): Initial time of integration. Defaults to 0.
            t_end (int, optional): Final time of integration. This parameter is also the max time of integration. 
            Defaults to 60.
            steps_integration (int, optional): Number of steps that will be used to split the integration time. 
            The higher this value is the better is the approximation of the solution. Defaults to 1000.

        Returns:
            array: Solution of the ODEs.
        """

        # event termination
        event_hit_target = self.hit_target
        event_hit_target.__dict__['terminal'] = True
        event_hit_target.__dict__['direction'] = -1

        # initilize the solver
        solver_ode = solve_ivp(self.equation_derivate, (t_start, t_end), u0, dense_output=True, events=(event_hit_target, self.max_height))
        
        # print some information about the calculation
        if verbose:
            print('Time to target = {:.2f} s'.format(solver_ode.t_events[0][0]))
            print('Time to highest point = {:.2f} s'.format(solver_ode.t_events[1][0]))

        # define grid of time points from 0 until impact time.
        t = np.linspace(0, solver_ode.t_events[0][0], steps_integration)
        # retrieve the solution for the time grid and plot the trajectory.
        results = solver_ode.sol(t)
        x, z = results[0], results[2]

        return x, z
