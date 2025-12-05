import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Contributors: SiQi Zhou, James Xu, Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # You can initialize more parameters here

        # Your code ends here

        ######################################################
        # Do not change these parameters below
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       target_acc = np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        mass = self._getURDFParameter('m')
        target_thrust, computed_target_rpy, pos_e, cur_rotation = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel,
                                                                         target_acc,
                                                                         mass=mass
                                                                         )
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )

        return rpm, pos_e, computed_target_rpy
    
    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel,
                               target_acc,
                               mass = 0.29
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.
        target_acc : ndarray
            (3,1)-shaped array of floats containing the desired acceleration.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.
        ndarray
            (3,3)-shaped array of floats representing the current rotation matrix (from quaternion).
        """

        #Write your code here
        # Evaluate desired thrust
        # Good og: Kp = 0.5*np.array([0.3, 0.3, 1]) 
        # solid for normal speed: 
        # Kp = 0.5*np.array([0.4, 0.4, 0.8]) 
        # Kd = np.array([0.2, 0.1, 0.3]) 

        Kp = np.array([0.4, 0.35, 0.6]) 
        Kd = np.array([0.2, 0.1, 0.3]) 

        # Kp = 0.5*np.array([1.5, 1.5, 1.2]) 
        # Kd = np.array([0.25, 0.25, 0.3]) 

        # for big speed
        # Kp = np.array([0.4, 0.4, 0.8]) 
        # Kd = 0.5*np.array([0.2, 0.1, 0.3]) 
        # Kd = np.array([0, 0, 0]) 
        e_r = target_pos - cur_pos # position error
        e_v = target_vel - cur_vel # velocity error 
        Fg = np.array([0, 0, self.GRAVITY])
        target_thrust = np.multiply(Kp, e_r) + np.multiply(Kd, e_v) + mass*target_acc + Fg
        # print(f"er, ev: {e_r, e_v}")
        # print(f"Fg: {Fg}")

        # scale thrust based on max tilt
        max_tilt = math.pi/4 # arbitrarily chosen
        xy_thrust = target_thrust[:2]
        horiz_max_thrust = target_thrust[2] * np.tan(max_tilt)
        thrust_ratio = horiz_max_thrust / np.linalg.norm(xy_thrust)
        if thrust_ratio < 1: # scale down
            target_thrust[0] *= thrust_ratio    
            target_thrust[1] *= thrust_ratio    

        # Evaluate desired orientation
        yaw_des = target_rpy[2]
        x_C_des = np.array([math.cos(yaw_des), math.sin(yaw_des), 0]) # desired heading direction 
        z_B_des = target_thrust / np.linalg.norm(target_thrust) # axis aligning with thrust direction 
        cross = np.cross(z_B_des, x_C_des)
        y_B_des = cross / np.linalg.norm(cross)
        x_B_des = np.cross(y_B_des, z_B_des)
        # print(f"axes: {x_B_des, y_B_des, z_B_des}")

        R_des = np.column_stack((x_B_des, y_B_des, z_B_des))
        # print(f"R_des: {R_des}")
        R_sci = Rotation.from_matrix(R_des)
        yaw, roll, pitch = R_sci.as_euler('zxy', False)
        target_rpy = np.array([roll, pitch, yaw])
        
        pos_e = e_r
        cur_rotation = np.reshape(p.getMatrixFromQuaternion(cur_quat), (3,3))
        # print(f"thrust: {target_thrust}")
        # print(f"target rpy: {target_rpy}")
        # print(f"pos e: {pos_e}")
        # print(f"cur rot: {cur_rotation}")

        #Your code ends here

        return target_thrust, target_rpy, pos_e, cur_rotation

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               cur_ang_vel,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_ang_vel : ndarray 
            (3,1)-shaped array of floats containing the current angular velocity.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """

        #Write your code here

        # Evaluate orientation
        roll, pitch, yaw = target_euler
        target_euler_zxy = np.array([yaw, roll, pitch])
        R_des = (Rotation.from_euler('zxy', target_euler_zxy, degrees=False)).as_matrix()
        R_cur = np.reshape(p.getMatrixFromQuaternion(cur_quat), (3,3))
        # print(f"R des: {R_des}")
        # print(f"R cur: {R_cur}")

        # Orientation error
        temp = np.dot(R_des.T, R_cur) - np.dot(R_cur.T, R_des)
        e_R = self.vee(temp)
        # print(f"e_R: {e_R}")

        # angular velocity error 
        e_w = target_rpy_rates - cur_ang_vel
        # print(f"e_w: {e_w}")

        # OG: 0.340
        # Kp = np.array([40000, 40000, 8000]) 
        # Kd = np.array([8000, 8000, 4000]) 

        # RMSE 0.341
        # Kp = np.array([50000, 50000, 10000])
        # Kd = np.array([10000, 10000, 5000])
        
        # RMSE 0.196
        # Kp = np.array([40000, 40000, 40000]) 
        # Kd = np.array([12000, 20000, 8000])
        
        # RMSE 0.196
        # Kp = np.array([35000, 38000, 45000]) 
        # Kd = np.array([3000, 5000, 3500])

        # RMSE 0.153
        # Kp = np.array([60000, 60000, 60000]) 
        # Kd = np.array([3000, 5000, 3500])

        # RMSE 0.1197
        Kp = np.array([100000, 110000, 100000]) 
        Kd = np.array([3000,   4000,   3500])

        # RMSE 0.1344 for normal time
        # Kp = np.array([220000, 400000, 200000]) 
        # Kd = np.array([12000,  20000,  15000])

        # RMSE 0.1208 for sped up
        # Kp = np.array([300000, 140000, 120000]) 
        # Kd = np.array([4000, 4500, 4600])

        # Kp = np.array([100000, 110000, 100000]) 
        # Kd = np.array([12000,  20000,  20000])

        # Kd = np.array([0, 0, 0]) 
        target_torques = -Kp*e_R + Kd*e_w
        # print(f"torques: {target_torques}")

        #Your code ends here

    ################################################################################

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()

    def vee(self, matrix):
        """utility function that evaluates the vee operator on a skew symmetric matrix.
        
        Parameters 
        ----------
        matrix : ndarray 
            (3,3)-shaped skew symmetric matrix 

        Returns
        -------
        ndarray 
            (3,1)-shaped array of floats 
        """
        u1 = matrix[2, 1]
        u2 = matrix[0, 2]
        u3 = matrix[1, 0]
        return np.array([u1, u2, u3]).transpose()
