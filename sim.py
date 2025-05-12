import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from pyMPC.mpc import MPCController
import scipy.sparse as sparse
import time

# --- Define initial condition and reference ---
x0 = 1.0
theta1_0 = np.pi / 2
theta2_0 = 0.1
y0 = [x0, theta1_0, theta2_0, 0.0, 0.0, 0.0]
wr = np.zeros(6)

# --- System Parameters ---
m0 = 1.0
m1 = 0.8
m2 = 0.5
L1 = 0.3
L2 = 0.45
g = 9.81

# Compute constants
d1 = m0 + m1 + m2
d2 = 0.5 * m1 * L1 + m2 * L1
d3 = 0.5 * m2 * L2
d4 = (1/3) * m1 * L1**2 + m2 * L1**2
d5 = 0.5 * m2 * L1 * L2
d6 = (1/3) * m2 * L2**2
f1 = (0.5 * m1 + m2) * L1 * g
f2 = 0.5 * m2 * L2 * g

# --- Control input function (constant force or state feedback placeholder) ---
control_input = lambda y: 0  # Replace with -K @ (y - wr) for feedback

# --- Dynamics function ---
def dipc_dynamics(t,y,u):
    x, theta1, theta2, dx, omega1, omega2 = y

    D = np.array([
        [d1, d2 * np.cos(theta1), d3 * np.cos(theta2)],
        [d2 * np.cos(theta1), d4, d5 * np.cos(theta1 - theta2)],
        [d3 * np.cos(theta2), d5 * np.cos(theta1 - theta2), d6]
    ])

    C = np.array([
        [0, -d2 * np.sin(theta1) * omega1, -d3 * np.sin(theta2) * omega2],
        [0, 0, d5 * np.sin(theta1 - theta2) * omega2],
        [0, -d5 * np.sin(theta1 - theta2) * omega1, 0]
    ])

    G_vec = np.array([
        0,
        -f1 * np.sin(theta1),
        -f2 * np.sin(theta2)
    ])

    H = np.array([1, 0, 0])
    # u = control_input(y)

    rhs = -C @ np.array([dx, omega1, omega2]) - G_vec + H * u
    accel = np.linalg.solve(D, rhs)

    dydt = [dx, omega1, omega2, accel[0], accel[1], accel[2]]
    return dydt

def drawcartpend_bw(y, m, M, L1, L2):
    plt.clf()
    plt.axis('equal')
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)
    
    x = y[0]
    theta1 = y[1]
    theta2 = y[2]

    # Cart dimensions
    cart_width = 0.5
    cart_height = 0.25
    wheel_radius = 0.05

    # --- Draw cart ---
    cart = plt.Rectangle((x - cart_width/2, -cart_height/2), cart_width, cart_height, color='black')
    plt.gca().add_patch(cart)

    # --- Draw wheels ---
    wheel1 = plt.Circle((x - cart_width/4, -cart_height/2 - wheel_radius), wheel_radius, color='black', fill=False)
    wheel2 = plt.Circle((x + cart_width/4, -cart_height/2 - wheel_radius), wheel_radius, color='black', fill=False)
    plt.gca().add_patch(wheel1)
    plt.gca().add_patch(wheel2)

    # --- First pendulum ---
    px1 = x + L1 * np.sin(theta1)
    py1 = L1 * np.cos(theta1)
    plt.plot([x, px1], [0, py1], 'r', linewidth=2)
    pend1_tip = plt.Circle((px1, py1), 0.1, color='red', fill=False)
    plt.gca().add_patch(pend1_tip)

    # --- Second pendulum ---
    px2 = px1 + L2 * np.sin(theta2)
    py2 = py1 + L2 * np.cos(theta2)
    plt.plot([px1, px2], [py1, py2], 'b', linewidth=2)
    pend2_tip = plt.Circle((px2, py2), 0.1, color='blue', fill=False)
    plt.gca().add_patch(pend2_tip)

    # --- Ground line ---
    ground_y = -cart_height/2 - wheel_radius - 0.05
    plt.plot([-5, 5], [ground_y, ground_y], 'k', linewidth=1)

    plt.pause(0.001)
    
# --- Simulation time ---
# t_eval = np.arange(0, 30, 0.01)

# # --- Solve ODE ---
# sol = solve_ivp(dipc_dynamics, [0, 30], y0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

# # --- Visualization setup ---
# plt.figure()

# # --- Draw function (reusing previously defined drawcartpend_bw) ---
# for k in range(len(sol.t)):
#     drawcartpend_bw(sol.y[:, k], m1, m0, L1, L2)

# plt.show()



Ts = 0.01
# Linearized System Matrices 
    
# Continuous-time system matrices, linearized about the upright, unstable equilibrium point 
Ac = np.array([
    [0,         0,         0,    1.0000,    0,         0],
    [0,         0,         0,    0,         1.0000,    0],
    [0,         0,         0,    0,         0,         1.0000],
    [0,   -9.6871,    1.0251,    0,         0,         0],
    [0,  118.3974, -35.8780,    0,         0,         0],
    [0,   -86.1072,  65.1499,    0,         0,         0]])

Bc = np.array([
    [0.0000],
    [0.0000],
    [0.0000],
    [0.8188],
    [-3.6585],
    [0.9292]
])

# Ac = np.array([
#     [0,         0,         0,    1.0000,    0,         0],
#     [0,         0,         0,    0,         1.0000,    0],
#     [0,         0,         0,    0,         0,         1.0000],
#     [0,   -12.7486,    0,    0,         0,         0],
#     [0,  75.1843, -20.4305,    0,         0,         0],
#     [0,   -75.1843,  55.8434,    0,         0,         0]])

# Bc = np.array([
#     [0.0000],
#     [0.0000],
#     [0.0000],
#     [1.0000],
#     [-3.3333],
#     [ 3.3333]
# ])

[nx, nu] = Bc.shape # number of states and number or inputs

# Simple forward euler discretization
Ad = np.eye(nx) + Ac*Ts
Bd = Bc*Ts



# MPC reference input and states (set-points)

xref = np.array([0.0, 0.0, 0.0, 0.0,0.0,0.0]) # reference state
uref = np.array([0.0])    # reference input
uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.

# Constraints
xmin = np.array([-100.0, -100, -100, -100,-100, -100])
xmax = np.array([100.0,   100.0, 100, 100, 100, 100])

umin = np.array([-20])
umax = np.array([20])

Dumin = np.array([-5])
Dumax = np.array([5])


# MPC cost function weights

Qx = sparse.diags([5, 10.0, 10.0, 1,1,1])   # Quadratic cost for states x0, x1, ..., x_N-1
QxN = sparse.diags([5, 10.0, 10.0, 1,1,1])  # Quadratic cost for xN
Qu = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
QDu = 0.01 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

# Initialize simulation system

# phi0 = 15*2*np.pi/360
x0 = np.array([  0.009999999776482582,
            0.009999999776482582,
            0.009999999776482582,
            0.0,
            0.0,
            0.0]) # initial state
t0 = 0
system_dyn = ode(dipc_dynamics).set_integrator('vode', method='bdf')
system_dyn.set_initial_value(x0, t0)
system_dyn.set_f_params(0.0)

Np = 100
Nc = 20
# Initialize and setup MPC controller

K = MPCController(Ad,Bd,Np=Np,Nc=Nc, x0=x0,xref=xref,uminus1=uminus1,
                  Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                  xmin=xmin,
                  xmax=xmax,
                  umin=umin,
                  umax=umax,
                  Dumin=Dumin,
                  Dumax=Dumax
                )
K.setup() # this initializes the QP problem for the first step

# Simulate in closed loop. Use MPC model as real system

# Simulate in closed loop
[nx, nu] = Bd.shape # number of states and number or inputs
len_sim = 2 # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)
xsim = np.zeros((nsim,nx))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts

time_start = time.time()

t_step = t0
uMPC = uminus1
for i in range(nsim):
    xsim[i,:] = system_dyn.y

    # MPC update and step. Could be in just one function call
    K.update(system_dyn.y, uMPC) # update with measurement
    print(system_dyn.y)
    print(uMPC)
    uMPC = K.output() # MPC step (u_k value)
    usim[i,:] = uMPC

    # System simulation
    system_dyn.set_f_params(uMPC) # set current input value
    system_dyn.integrate(t_step + Ts)

    # Time update
    t_step += Ts
time_sim = time.time() - time_start


# for k in range(len(tsim)):
#     drawcartpend_bw(xsim[k,:], m1, m0, L1, L2)

# plt.show()
# # Plot results

# fig,axes = plt.subplots(3,1, figsize=(10,10))

# axes[0].plot(tsim, xsim[:,0], "k", label='p')
# axes[0].plot(tsim, xref[0]*np.ones(np.shape(tsim)), "r--", label="p_ref")
# axes[0].set_title("Position (m)")

# axes[1].plot(tsim, xsim[:,2]*360/2/np.pi, label="phi")
# axes[1].plot(tsim, xref[2]*360/2/np.pi*np.ones(np.shape(tsim)), "r--", label="phi_ref")
# axes[1].set_title("Angle (deg)")

# axes[2].plot(tsim, usim[:,0], label="u")
# axes[2].plot(tsim, uref*np.ones(np.shape(tsim)), "r--", label="u_ref")
# axes[2].set_title("Force (N)")

# for ax in axes:
#     ax.grid(True)
#     ax.legend()
    
# plt.show()