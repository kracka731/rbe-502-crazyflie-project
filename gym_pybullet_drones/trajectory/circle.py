import numpy as np
from diamond import M


def circle(t, tf=8):
    """
    Generate the desired state of a drone following a circular trajectory.

    The function computes the drone’s position, velocity, and acceleration
    at a given time 't' while following a circular trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tf (float): Total trajectory duration.

    Returns:
        desired_state (dict):
            - 'pos'   (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'   (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'   (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk'  (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'   (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """

    """
    Write your code here.
    """
    time_per_phase = 5
    n_samples = int(time_per_phase/0.1)

    M_t0 = M(0)
    M_t1 = M(time_per_phase)
    A = np.vstack((M_t0, M_t1))

    # Phase 1 ----------------------------------------------
    bMat = phase_1_bMat()  # 6x3 matrix
    a = np.linalg.inv(A) @ bMat  # 6x1

    time_arr = np.linspace(0, time_per_phase, n_samples)
    p1_P_t = np.empty((1, 3, 3))
    for t in time_arr:
        M_t = M(t)
        P_t = M_t @ a  # 3x3 matrix
        # store as slices in 3D matrix
        p1_P_t = np.concatenate((p1_P_t, P_t.reshape(1, 3, 3)), axis=0)

    pos = p1_P_t[1:, 0, :]
    vel = p1_P_t[1:, 1, :]
    acc = p1_P_t[1:, 2, :]

    # Phase 2 -----------------------------------------------
    R = 1  # m
    # omega = 2*np.pi/time_per_phase
    z_d = 1  # m

    th0 = w0 = wd0 = wf = wdf = 0
    thf = 2*np.pi
    b = np.vstack((th0, w0, wd0, thf, wf, wdf))  # 8x1
    Ma_t1 = M(time_per_phase)
    Ma_t2 = M(time_per_phase*2)
    A = np.vstack((Ma_t1, Ma_t2))  # 8x8
    a = np.linalg.inv(A) @ b  # 8x8 * 8x1 = 8x1
    # print(f"shape of A: {np.shape(A)}, shape of b: {np.shape(b)}, a shape: {np.shape(a)}")

    angular_P_t = np.empty((3, 1))
    for t in time_arr:
        M_t = M(t + time_per_phase)  # 4x8
        P_t = M_t @ a  # 4x1
        angular_P_t = np.hstack((angular_P_t, P_t))

    p2_P_t = np.empty((1, 3, 3))
    for i in range(len(time_arr)):
        # Extract angular motion info
        t = time_arr[i] + time_per_phase
        w = angular_P_t[0, i] / t  # access ang vel
        w_dot = angular_P_t[1, i]
        w_ddot = angular_P_t[2, i]

        # utils
        w_t = w*t
        wd_t = w_dot*t
        wdd_t = w_ddot*t
        wd_t_w = w_dot  # wd_t + w   #
        temp = w_ddot  # wdd_t + 2*w_dot  #

        # xyz pos
        r = np.array([[R*np.cos(w_t)], [R*np.sin(w_t)], [z_d]])

        # xyz vel
        v_x = -R * (wd_t_w) * np.sin(w_t)
        v_y =  R * (wd_t_w) * np.cos(w_t)
        v_z = 0

        # xyz acc
        a_x = -R * temp * np.sin(w_t) - R * wd_t_w**2 * np.cos(w_t)
        a_y =  R * temp * np.cos(w_t) - R * wd_t_w**2 * np.sin(w_t)
        a_z = 0

        P_t = np.vstack((r.T, [v_x, v_y, v_z], [a_x, a_y, a_z]))
        p2_P_t = np.concatenate((p2_P_t, P_t.reshape(1, 3, 3)), axis=0)

    pos = np.concatenate((pos, p2_P_t[1:, 0, :]))
    vel = np.concatenate((vel, p2_P_t[1:, 1, :]))
    acc = np.concatenate((acc, p2_P_t[1:, 2, :]))

    desired_state = {
        'pos': pos,
        'vel': vel,
        'acc': acc,
        'jerk': np.array([0, 0, 0]),
        'yaw': 0,
        'yawdot': 0
    }

    return desired_state


# Helper Functions


def phase_1_bMat() -> np.ndarray:
    """
    Finds the b matrix for Phase 1, when the quadrotor moves to the start 
    position of the circle.
    Returns:
        b (ndarray):
            (6,3)-shaped array of floats where each row represents starting
            or ending pos, vel, acc.
    """
    # positions in the world frame
    r0 = np.array([0, 0, 0.5])
    r1 = np.array([1, 0, 1])

    # trajectory halts at the start/end of its trajectory
    v0 = vf = a0 = af = np.array([0, 0, 0])

    b = np.vstack([r0, v0, a0, r1, vf, af])
    # print(f"b: {b}")
    return b


def phase_2_bMat() -> np.ndarray:
    """
    Finds the four vertices of a diamond. 
    Returns:
        M (ndarray):
            (3,4)-shaped array of floats representing the four vertices
            of a diamond in 3D space.
    """
    # position in the world frame
    r1 = np.array([1, 0, 1])

    # trajectory halts at the start/end of its trajectory
    v0 = vf = a0 = af = np.array([0, 0, 0])

    b = np.vstack([r1, v0, a0, r1, vf, af])
    return b


def phase_3_bMat() -> np.ndarray:
    """
    Finds the b matrix for Phase 3, when the quadrotor moves to the start 
    position of the circle.
    Returns:
        b (ndarray):
            (6,3)-shaped array of floats where each row represents starting
            or ending pos, vel, acc.
    """
    # positions in the world frame
    r0 = np.array([0, 0, 0.5])
    r1 = np.array([1, 0, 1])

    # trajectory halts at the start/end of its trajectory
    v0 = vf = a0 = af = np.array([0, 0, 0])

    b = np.vstack([r1, v0, a0, r0, vf, af])
    # print(f"b: {b}")
    return b


def M_a(t):
    """M, but also accounts for jerk"""
    M = [[1, t, t**2, t**3,     t**4,      t**5,       t**6,       t**7],
         [0, 1, 2*t,  3*(t**2), 4*(t**3),  5*(t**4),   6*(t**5),   7*(t**6)],
         [0, 0, 2,    6*t,     12*(t**2), 20*(t**3),  30*(t**4),  42*(t**5)],
         [0, 0, 0,    6,       24*t,      60*(t**2), 120*(t**3), 210*(t**4)]]
    return np.asarray(M)
