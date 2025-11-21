import matplotlib.pyplot as plt
import numpy as np
from circle import circle
from diamond import diamond


def main():
    state = circle(0, 15)
    n_samples = int(5/0.25)
    graph(state, 0, 5, n_samples, "Circle")


def graph(state, t0, tf, n_samples, shape: str):
    time = np.linspace(t0, tf, n_samples)
    pos = state['pos']
    vel = state['vel']
    acc = state['acc']

    # XYZ Position Graph
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f"{shape} Trajectory in XYZ Space")

    # bx = fig.add_subplot()
    f, (pp, pv, pa) = plt.subplots(3, 1)
    # f, px = plt.subplots()

    # x(t), y(t), z(t) graphs
    pp.plot(time, pos[:, 0], label='X Position')
    pp.plot(time, pos[:, 1], label='Y Position')
    pp.plot(time, pos[:, 2], label='Z Position')
    pp.set_xlabel('Time (s)')
    pp.set_ylabel('Position (m)')
    pp.legend()
    pp.set_title(f"Position for {shape} Trajectory over Time")

    # x_dot(t), y_dot(t), z_dot(t) graphs
    pv.plot(time, vel[:, 0], label='X Velocity')
    pv.plot(time, vel[:, 1], label='Y Velocity')
    pv.plot(time, vel[:, 2], label='Z Velocity')
    pv.set_xlabel('Time (s)')
    pv.set_ylabel('Velocity (m/s)')
    pv.legend()
    pv.set_title(f"Velocity for {shape} Trajectory over Time")

    # x_ddot(t), y_ddot(t), z_ddot(t) graphs
    pa.plot(time, acc[:, 0], label='X Acceleration')
    pa.plot(time, acc[:, 1], label='Y Acceleration')
    pa.plot(time, acc[:, 2], label='Z Acceleration')
    pa.set_xlabel('Time (s)')
    pa.set_ylabel('Acceleration (m/s^2)')
    pa.legend()
    pa.set_title(f"Acceleration for {shape} Trajectory over Time")

    plt.show()


if __name__ == "__main__":
    main()
