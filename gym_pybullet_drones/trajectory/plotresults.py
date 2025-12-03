import matplotlib.pyplot as plt
import numpy as np
from circle import circle
from diamond import diamond


def main():
    # Circle Trajectory
    tf = 15
    state, time_arr = generate_whole_trajectory(tf, circle)
    graph(state, time_arr, "Circle")

    # Diamond Trajectory
    tf = 8
    state, time_arr = generate_whole_trajectory(tf, diamond)
    graph(state, time_arr, "Diamond")
    plt.show()


def generate_whole_trajectory(tf: float, shape:circle):
    n_samples = int(tf/0.1)
    time_arr = np.linspace(0, tf, n_samples)

    all_pos, all_vel, all_acc = np.empty((3, 1)), np.empty((3, 1)), np.empty((3, 1))
    for t in time_arr:
        state = shape(t, tf)
        pos = state['pos']
        vel = state['vel']
        acc = state['acc']
        all_pos = np.hstack((all_pos, pos.reshape((3, 1))))
        all_vel = np.hstack((all_vel, vel.reshape((3, 1))))
        all_acc = np.hstack((all_acc, acc.reshape((3, 1))))
    state = {
        'pos': all_pos[:, 1:],
        'vel': all_vel[:, 1:],
        'acc': all_acc[:, 1:]
    }
    return state, time_arr

def graph(state, time: np.ndarray, shape: str):
    pos = state['pos']
    vel = state['vel']
    acc = state['acc']

    # XYZ Position Graph
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos[0, :], pos[1, :], pos[2, :])

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f"{shape} Trajectory in XYZ Space")

    f, (pp, pv, pa) = plt.subplots(3, 1)

    # x(t), y(t), z(t) graphs
    pp.plot(time, pos[0, :], label='X Position')
    pp.plot(time, pos[1, :], label='Y Position')
    pp.plot(time, pos[2, :], label='Z Position')
    pp.set_xlabel('Time (s)')
    pp.set_ylabel('Position (m)')
    pp.legend()
    pp.set_title(f"Position for {shape} Trajectory over Time")

    # x_dot(t), y_dot(t), z_dot(t) graphs
    pv.plot(time, vel[0, :], label='X Velocity')
    pv.plot(time, vel[1, :], label='Y Velocity')
    pv.plot(time, vel[2, :], label='Z Velocity')
    pv.set_xlabel('Time (s)')
    pv.set_ylabel('Velocity (m/s)')
    pv.legend()
    pv.set_title(f"Velocity for {shape} Trajectory over Time")

    # x_ddot(t), y_ddot(t), z_ddot(t) graphs
    pa.plot(time, acc[0, :], label='X Acceleration')
    pa.plot(time, acc[1, :], label='Y Acceleration')
    pa.plot(time, acc[2, :], label='Z Acceleration')
    pa.set_xlabel('Time (s)')
    pa.set_ylabel('Acceleration (m/s^2)')
    pa.legend()
    pa.set_title(f"Acceleration for {shape} Trajectory over Time")


if __name__ == "__main__":
    main()
