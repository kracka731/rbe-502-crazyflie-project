import numpy as np


def diamond(t, tfinal=8):
    """
    Generate the desired state of a drone following a diamond-shaped
    trajectory.

    The function computes the drone’s position, velocity, and acceleration at
    any given time 't' while following a diamond-shaped trajectory.

    Parameters:
        t (float): Current time (in seconds).
        tfinal (float): Total trajectory duration.

    Returns:
        desired_state (dict):
            - 'pos'  (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'  (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'  (np.ndarray, shape (3,)): Desired acceleration [ax,ay,az].
            - 'jerk' (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'  (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """
    """
    Write your code here.
    """
    vertices = find_diamond_vertices()
    v0 = vf = a0 = af = np.array([0, 0, 0])

    # Split each side of the diamond evenly per its 4 sides
    t_per_side = tfinal/4
    side = int(np.floor(t / t_per_side))
    if side == 4:
        side = 3
    
    t0 = t_per_side*side
    # Form constraint matrix
    M_t0 = M(t0)
    M_tf = M(t_per_side*(side+1))
    A = np.vstack((M_t0, M_tf))

    p0 = vertices[side, :]
    pf = vertices[side+1, :]
    b = np.array([p0, v0, a0, pf, vf, af])  # 6x3 matrix
    a = np.linalg.inv(A) @ b  # 6x1 coefficient matrix

    # Evaluate pos, vel, acc
    M_t = M(t)
    P_t = M_t @ a  # 3x3 matrix

    # Extract data & convert to column vectors
    pos = np.reshape(P_t[0, :], (3, 1))
    vel = np.reshape(P_t[1, :], (3, 1))
    acc = np.reshape(P_t[2, :], (3, 1))

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


def find_diamond_vertices() -> np.ndarray:
    """
    Finds the four vertices of a diamond.
    Returns:
        M (ndarray):
            (5,3)-shaped array of floats representing the four vertices of
            a diamond in 3D space.
    """
    u = 1/np.sqrt(2)  # unit length, length of one diamond side
    ux = 1

    p0 = np.array([0, 0, 0])
    p1 = np.array([ux, u, u])
    p2 = np.array([ux, 0, 2*u])
    p3 = np.array([ux, -u, u])
    p4 = p0

    vertices = np.vstack((p0, p1, p2, p3, p4))
    return vertices


def M(t):
    """
    Based on a time t, evaluate part of the coefficient matrix, up to
    a 2nd derivative

    Parameters:
        t(float): time (in seconds)

    Returns:
        M (ndarray):
            A 3x6 array
    """
    M = [[1, t, t**2, t**3,     t**4,      t**5,],
         [0, 1, 2*t,  3*(t**2), 4*(t**3),  5*(t**4)],
         [0, 0, 2,    6*t,     12*(t**2), 20*(t**3)]]
    return np.asarray(M)
