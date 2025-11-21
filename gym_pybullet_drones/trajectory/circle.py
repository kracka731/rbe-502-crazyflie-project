import numpy as np
from diamond import eval_coeff_matrix


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
    n_samples = int(5/0.25)
    M_t0 = eval_coeff_matrix(0)
    M_tf = eval_coeff_matrix(5)
    A = np.vstack((M_t0, M_tf))
    bMat = phase_1_bMat()  # 6x3 matrix
    coefficients = np.linalg.inv(A) @ bMat  # 6x3

    array = np.linspace(0, 5, n_samples)
    p1_P_t = np.empty((1, 3, 3))
    for t in array:
        M_t = eval_coeff_matrix(t)
        P_t = M_t @ coefficients
        p1_P_t = np.concatenate((p1_P_t, P_t.reshape(1, 3, 3)), axis=0)

    pos = p1_P_t[1:, 0, :]
    vel = p1_P_t[1:, 1, :]
    acc = p1_P_t[1:, 2, :]

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
    r0 = np.array([0, 0, 0.5])
    r1 = np.array([1, 0, 1])

    R = 1  # m
    omega = 2
    z_d = 1  # m
    r_circle = np.array([R*np.cos(omega*t), R*np.sin(omega*t), z_d])

    b = np.vstack([r0, v0, a0, r1, vf, af])
    return b

