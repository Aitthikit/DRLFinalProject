import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from pyMPC.mpc import MPCController
import scipy.sparse as sparse
import time


class MPC():
    def __init__(self):  
    # --- Define initial condition and reference ---
        # self.x0 = 1.0
        # self.theta1_0 = np.pi / 2
        # self.theta2_0 = 0.1
        self.y0 = [0, 0, 0, 0.0, 0.0, 0.0]
        self.wr = np.zeros(6)
        # --- System Parameters ---
        self.m0 = 1.0
        self.m1 = 0.8
        self.m2 = 0.5
        self.L1 = 0.3
        self.L2 = 0.45
        self.g = 9.81
        # Compute constants
        self.d1 = self.m0 + self.m1 + self.m2
        self.d2 = 0.5 * self.m1 * self.L1 + self.m2 * self.L1
        self.d3 = 0.5 * self.m2 * self.L2
        self.d4 = (1/3) * self.m1 * self.L1**2 + self.m2 * self.L1**2
        self.d5 = 0.5 * self.m2 * self.L1 * self.L2
        self.d6 = (1/3) * self.m2 * self.L2**2
        self.f1 = (0.5 * self.m1 + self.m2) * self.L1 * self.g
        self.f2 = 0.5 * self.m2 * self.L2 * self.g

        # --- Control input function (constant force or state feedback placeholder) ---
        self.control_input = lambda y: 0  # Replace with -K @ (y - wr) for feedback

        self.Ts = 0.01
        # Linearized System Matrices 
        
        # Continuous-time system matrices, linearized about the upright, unstable equilibrium point 
        # self.Ac = np.array([
        #     [0,         0,         0,    1.0000,    0,         0],
        #     [0,         0,         0,    0,         1.0000,    0],
        #     [0,         0,         0,    0,         0,         1.0000],
        #     [0,   -9.6871,    1.0251,    0,         0,         0],
        #     [0,  118.3974, -35.8780,    0,         0,         0],
        #     [0,   -86.1072,  65.1499,    0,         0,         0]])

        # self.Bc = np.array([
        #     [0.0000],
        #     [0.0000],
        #     [0.0000],
        #     [0.8188],
        #     [-3.6585],
        #     [0.9292]
        # ])

        self.Ac = np.array([
            [0,         0,         0,    1.0000,    0,         0],
            [0,         0,         0,    0,         1.0000,    0],
            [0,         0,         0,    0,         0,         1.0000],
            [0,   -12.7486,    0,    0,         0,         0],
            [0,  75.1843, -20.4305,    0,         0,         0],
            [0,   -75.1843,  55.8434,    0,         0,         0]])

        self.Bc = np.array([
            [0.0000],
            [0.0000],
            [0.0000],
            [1.0000],
            [-3.3333],
            [ 3.3333]
        ])
        [self.nx, self.nu] = self.Bc.shape # number of states and number or inputs

        # Simple forward euler discretization
        self.Ad = np.eye(self.nx) + self.Ac*self.Ts
        self.Bd = self.Bc*self.Ts

        

    # --- Dynamics function ---
    def dipc_dynamics(self,t,y,u):
        x, theta1, theta2, dx, omega1, omega2 = y

        D = np.array([
            [self.d1, self.d2 * np.cos(theta1), self.d3 * np.cos(theta2)],
            [self.d2 * np.cos(theta1), self.d4, self.d5 * np.cos(theta1 - theta2)],
            [self.d3 * np.cos(theta2), self.d5 * np.cos(theta1 - theta2), self.d6]
        ])

        C = np.array([
            [0, -self.d2 * np.sin(theta1) * omega1, -self.d3 * np.sin(theta2) * omega2],
            [0, 0, self.d5 * np.sin(theta1 - theta2) * omega2],
            [0, -self.d5 * np.sin(theta1 - theta2) * omega1, 0]
        ])

        G_vec = np.array([
            0,
            -self.f1 * np.sin(theta1),
            -self.f2 * np.sin(theta2)
        ])

        H = np.array([1, 0, 0])
        # u = control_input(y)

        rhs = -C @ np.array([dx, omega1, omega2]) - G_vec + H * u
        accel = np.linalg.solve(D, rhs)

        dydt = [dx, omega1, omega2, accel[0], accel[1], accel[2]]
        return dydt


    # def getMPC(self,xref,cost,init,predictH,controlH):
    def getMPC(self,x0):
        # MPC reference input and states (set-points)

        xref = np.array([0.0, 0.0, 0.0, 0.0,0.0,0.0]) # reference state
        uref = np.array([0.0])    # reference input
        uminus1 = np.array([0.0])     # input at time step negative one - used to penalize the first delta u at time instant 0. Could be the same as uref.
        # Constraints
        xmin = np.array([-100.0, -100, -100, -100,-100, -100])
        xmax = np.array([100.0,   100.0, 100, 100, 100, 100])

        umin = np.array([-25])
        umax = np.array([25])

        Dumin = np.array([-5])
        Dumax = np.array([5])

        # MPC cost function weights

        # Qx = sparse.diags([5, 10.0, 10.0, 3,1,1])   # Quadratic cost for states x0, x1, ..., x_N-1
        # QxN = sparse.diags([5, 10.0, 10.0, 3,1,1])  # Quadratic cost for xN
        Qx = sparse.diags([5, 10.0, 10.0, 1,1,1])   # Quadratic cost for states x0, x1, ..., x_N-1
        QxN = sparse.diags([5, 10.0, 10.0, 1,1,1])  # Quadratic cost for xN
        Qu = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
        QDu = 0.01 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

        # Initialize simulation system

        # phi0 = 15*2*np.pi/360
        x0 = x0 # initial state
        # system_dyn = ode(dipc_dynamics).set_integrator('vode', method='bdf')
        # system_dyn.set_initial_value(x0, t0)
        # system_dyn.set_f_params(0.0)

        Np = 100
        Nc = 20
        # Np = 150
        # Nc = 90
        # Initialize and setup MPC controller

        K = MPCController(self.Ad,self.Bd,Np=Np,Nc=Nc, x0=x0,xref=xref,uminus1=uminus1,
                        Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                        xmin=xmin,
                        xmax=xmax,
                        umin=umin,
                        umax=umax,
                        # Dumin=Dumin,
                        # Dumax=Dumax
                        )
        K.setup() # this initializes the QP problem for the first step

        return K

    # Simulate in closed loop. Use MPC model as real system

    # Simulate in closed loop
    # [nx, nu] = Bd.shape # number of states and number or inputs
    # len_sim = 10 # simulation length (s)
    # nsim = int(len_sim/Ts) # simulation length(timesteps)
    # xsim = np.zeros((nsim,nx))
    # usim = np.zeros((nsim,nu))
    # tsim = np.arange(0,nsim)*Ts

    # time_start = time.time()

    # t_step = t0
    # uMPC = uminus1
    # for i in range(nsim):
    #     xsim[i,:] = system_dyn.y

    #     # MPC update and step. Could be in just one function call
    #     K.update(system_dyn.y, uMPC) # update with measurement
    #     uMPC = K.output() # MPC step (u_k value)
    #     usim[i,:] = uMPC

    #     # System simulation
    #     system_dyn.set_f_params(uMPC) # set current input value
    #     system_dyn.integrate(t_step + Ts)

    #     # Time update
    #     t_step += Ts
    # time_sim = time.time() - time_start