import matplotlib.pyplot as plt
import numpy as np
from motion_equations import Quadcopter6DOF

quadd = Quadcopter6DOF()

m = 0.1   #kg
g = 9.81  #m/s/s
dx=0.114, 
dy=0.0825

# Plot Thrust as a function of RPM for various vertical velocity conditions
RPM = np.linspace(1000,6000,200)
vertvel = np.array([0,0,1] + 9*[0])
Thrust_m2vel = np.array([quadd.thrust(x=2*vertvel, u=rpmIn, dx=dx, dy=dy) for rpmIn in RPM])
Thrust_m1vel = np.array([quadd.thrust(1*vertvel,rpmIn, dx=dx, dy=dy) for rpmIn in RPM])
Thrust_0vel  = np.array([quadd.thrust(0*vertvel,rpmIn, dx=dx, dy=dy) for rpmIn in RPM])
Thrust_p1vel = np.array([quadd.thrust(-1*vertvel,rpmIn, dx=dx, dy=dy) for rpmIn in RPM])
Thrust_p2vel = np.array([quadd.thrust(-2*vertvel,rpmIn, dx=dx, dy=dy) for rpmIn in RPM])

fig = plt.figure(figsize=(8,8))
plt.plot(RPM, 4 * Thrust_m2vel / (m*g) )
plt.plot(RPM, 4 * Thrust_m1vel / (m*g) )
plt.plot(RPM, 4 * Thrust_0vel / (m*g) )
plt.plot(RPM, 4 * Thrust_p1vel / (m*g) )
plt.plot(RPM, 4 * Thrust_p2vel / (m*g) )
plt.plot(RPM, np.ones(np.size(RPM)), 'k--')
plt.legend(('Airspeed = -2 m/s','Airpseed = -1 m/s','Airspeed =  0 m/s', \
            'Airpseed =  1 m/s','Airspeed =  2 m/s'), loc='upper left')
plt.xlabel('Propeller RPM (x4)')
plt.ylabel('Thrust (g)')
plt.title('Quadcopter Thrust for different Vertical Airspeeds')
plt.show()