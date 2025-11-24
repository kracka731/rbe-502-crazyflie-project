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
    z_d = 1  # m

    th0 = w0 = wd0 = wf = wdf = 0
    thf = 2*np.pi
    b = np.vstack((th0, w0, wd0, thf, wf, wdf))  # 6x1
    M_t1 = M(time_per_phase)
    M_t2 = M(time_per_phase*2)
    A = np.vstack((M_t1, M_t2))  # 6x6
    a = np.linalg.inv(A) @ b  # 6x6 * 6x1 = 6x1

    angular_P_t = np.empty((3, 1))
    for t in time_arr:
        M_t = M(t + time_per_phase)  # 3x6
        P_t = M_t @ a  # 3x1
        angular_P_t = np.hstack((angular_P_t, P_t))

    p2_P_t = np.empty((1, 3, 3))
    for i in range(len(time_arr)):
        # Extract angular motion info
        t = time_arr[i] + time_per_phase
        w = angular_P_t[0, i] / t
        w_dot = angular_P_t[1, i]
        w_ddot = angular_P_t[2, i]
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
        p2_P_t = np.concatenate((p2_P_t, P_t.reshape(1, 3, 3)), axis=0)

    pos = np.concatenate((pos, p2_P_t[1:, 0, :]))
    vel = np.concatenate((vel, p2_P_t[1:, 1, :]))
    acc = np.concatenate((acc, p2_P_t[1:, 2, :]))

    # Phase 3 ----------------------------------------------
    M_t2 = M(time_per_phase*2)
    M_t3 = M(time_per_phase*3)
    A = np.vstack((M_t2, M_t3))
    b = phase_3_bMat()  # 6x3 matrix
    a = np.linalg.inv(A) @ b  # 6x1

    p3_P_t = np.empty((1, 3, 3))
    for t in time_arr:
        M_t = M(t+(2*time_per_phase))
        P_t = M_t @ a  # 3x3 matrix
        # store as slices in 3D matrix
        p3_P_t = np.concatenate((p3_P_t, P_t.reshape(1, 3, 3)), axis=0)

    pos = np.concatenate((pos, p3_P_t[1:, 0, :]))
    vel = np.concatenate((vel, p3_P_t[1:, 1, :]))
    acc = np.concatenate((acc, p3_P_t[1:, 2, :]))

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
