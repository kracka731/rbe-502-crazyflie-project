import numpy as np
from gym_pybullet_drones.trajectory.diamond import M


def circle(t: float, tf: float = 8):
    """
    Generate the desired state of a drone following a circular trajectory.

    The function computes the drone’s position, velocity, and acceleration
    at a given time 't' while following a circular trajectory.

    Parameters:
        t (float): Current time step (in seconds).
        tf (float): Total trajectory duration.

    Returns:
        desired_state (dict):
            - 'pos'  (np.ndarray, shape (3,)): Desired position [x, y, z].
            - 'vel'  (np.ndarray, shape (3,)): Desired velocity [vx, vy, vz].
            - 'acc'  (np.ndarray, shape (3,)): Desired acceleration [ax, ay, az].
            - 'jerk' (np.ndarray, shape (3,)): Desired jerk (set to zero).
            - 'yaw'  (float): Desired yaw angle (set to zero).
            - 'yawdot' (float): Desired yaw rate (set to zero).
    """

    """
    Write your code here.
    """
    time_per_phase = tf/3.0
    phase_num = int(np.ceil(t / time_per_phase))
    if phase_num == 0:
        phase_num = 1

    P_t = np.empty((3,3))
    # Set up coefficient matricies
    M_t0 = M(time_per_phase*(phase_num-1))
    M_tf = M(time_per_phase*phase_num)
    A = np.vstack((M_t0, M_tf))

    match phase_num:
        case 1:  # Phase 1 ----------------------------------------------
            bMat = phase_1_bMat()  # 6x3 matrix
            a = np.linalg.inv(A) @ bMat  # 6x1

            M_t = M(t)
            P_t = M_t @ a  # 3x3 matrix
            
        case 2: # Phase 2 -----------------------------------------------
            R = 1  # m
            z_d = 1  # m

            th0 = w0 = wd0 = wf = wdf = 0
            thf = 2*np.pi
            b = np.vstack((th0, w0, wd0, thf, wf, wdf))  # 6x1
            a = np.linalg.inv(A) @ b  # 6x6 * 6x1 = 6x1

            M_t = M(t)  # 3x6
            angular_P_t = M_t @ a  # 3x1
            w = angular_P_t[0, 0] / t
            # print(f"omega: {w}, ang_pt = {angular_P_t}")
            w_dot = angular_P_t[1, 0]
            w_ddot = angular_P_t[2, 0]
            w_t = w*t

            # xyz pos
            r = np.array([[R*np.cos(w_t)], [R*np.sin(w_t)], [z_d]])

            # xyz vel
            v_x = -R * (w_dot) * np.sin(w_t)
            v_y = R * (w_dot) * np.cos(w_t)
            v_z = 0

            # xyz acc
            a_x = -R * w_ddot * np.sin(w_t) - R * w_dot**2 * np.cos(w_t)
            a_y = R * w_ddot * np.cos(w_t) - R * w_dot**2 * np.sin(w_t)
            a_z = 0

            P_t = np.vstack((r.T, [v_x, v_y, v_z], [a_x, a_y, a_z]))

        case 3:  # Phase 3 ----------------------------------------------
            b = phase_3_bMat()  # 6x3 matrix
            a = np.linalg.inv(A) @ b  # 6x1

            M_t = M(t)
            P_t = M_t @ a  # 3x3 matrix

        case _:
            print(f"something has gone wrong. this is presumed to be phase {phase_num}")

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
    return b


def phase_2_bMat() -> np.ndarray:
    """
    Finds the b matrix for Phase 2, when the quadrotor moves around the circle
    from/to the same point.
    Returns:
        b (ndarray):
            (6,3)-shaped array of floats where each row represents starting
            or ending pos, vel, acc.
    """
    # position in the world frame
    r1 = np.array([1, 0, 1])

    # trajectory halts at the start/end of its trajectory
    v0 = vf = a0 = af = np.array([0, 0, 0])

    b = np.vstack([r1, v0, a0, r1, vf, af])
    return b


def phase_3_bMat() -> np.ndarray:
    """
    Finds the b matrix for Phase 3, when the quadrotor moves from the start
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
    return b
