# Feedback‐linearized hover controller: read poses[1] instead of poses[0]

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler_scipy(w, x, y, z):
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=False)

class PositionController(Node):
    def __init__(self):
        super().__init__('FL_Position_Controller')
        # SDF params
        self.m, self.g = 1.5, 9.81
        self.Ixx, self.Iyy, self.Izz = 0.0347563, 0.0458929, 0.0977
        self.c_T, self.c_M, self.L = 8.54858e-06, 0.06, 0.244
        self.max_rot_vel = 1100.0

        # hover setpoint
        self.des_z = 2.0

        #try adding references for phi and theta to be zero

        # PD gains
        self.kp_z,   self.kd_z   = 2.0, 1.0
        self.kp_phi, self.kd_phi = 8.0, 2.0
        self.kp_theta, self.kd_theta = 8.0, 2.0
        self.kp_psi, self.kd_psi = 4.0, 1.0

        # history for rate estimates
        self.last_time = self.get_clock().now()
        self.last_z = self.last_phi = self.last_theta = self.last_psi = None

        # state
        self.z = 0.0
        self.error_z = 0.0
        self.phi = self.theta = self.psi = 0.0

        # subscriptions / pubs
        self.create_subscription(
            PoseArray,
            '/world/quadcopter/pose/info',
            self.pose_callback,
            10
        )
        self.pub = self.create_publisher(
            Actuators,
            '/X3/gazebo/command/motor_speed',
            10
        )

    def pose_callback(self, msg: PoseArray):
        n = len(msg.poses)
        if n < 2:
            self.get_logger().warn(f"Expected ≥2 poses, got {n}")
            return

        # debug: print both entries to find which is nonzero
        for i, p in enumerate(msg.poses):
            self.get_logger().debug(f"Pose[{i}] z={p.position.z:.2f}")

        # take the second entry (index 1)
        p = msg.poses[1].position
        o = msg.poses[1].orientation
        self.get_logger().info(f"Using Pose[1] z={p.z:.2f}")

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 1e-4:
            return

        # update state
        self.z = p.z
        self.phi, self.theta, self.psi = quaternion_to_euler_scipy(
            o.w, o.x, o.y, o.z
        )

        # estimate rates
        if self.last_z is None:
            zdot = phidot = thetadot = psidot = 0.0
        else:
            error_z_dot = (self.error_z - self.last_error_z) / dt
            zdot     = (self.z     - self.last_z)     / dt
            phidot   = (self.phi   - self.last_phi)   / dt
            thetadot = (self.theta - self.last_theta) / dt
            psidot   = (self.psi   - self.last_psi)   / dt
        # virtual inputs

        #Either control x,y with PID or set them to zero

        #try changing the error to the derivative of the z, then for phi,theta and psi
        error_z = (self.des_z - self.z)
        v_z     =  self.kp_z*error_z   - self.kd_z* zdot
        v_phi   = -self.kp_phi*self.phi             - self.kd_phi* phidot
        v_theta = -self.kp_theta*self.theta         - self.kd_theta* thetadot
        v_psi   =  self.kp_psi*(0.0 - self.psi)      - self.kd_psi* psidot

        # feedback‐linearizing control law
        u0 = self.m*(v_z + self.g)/(np.cos(self.phi)*np.cos(self.theta))
        u0 = float(np.clip(u0, 0.5*self.m*self.g, 2.0*self.m*self.g))
        u1 = -(self.Iyy - self.Izz)*thetadot*psidot + self.Ixx*v_phi
        u2 = -(self.Izz - self.Ixx)*phidot*psidot   + self.Iyy*v_theta
        u3 = -(self.Ixx - self.Iyy)*phidot*thetadot + self.Izz*v_psi

        # map to individual thrusts
        A = np.array([[1,1,1,1],
                      [0,1,0,-1],
                      [-1,0,1,0],
                      [1,-1,1,-1]])
        b = np.array([u0,
                      u1/self.L,
                      u2/self.L,
                      u3*(self.c_T/self.c_M)])
        T = np.linalg.solve(A, b)

        # rotor speeds
        ω = np.sqrt(np.maximum(T,0)/self.c_T)
        ω_cmd = np.clip(ω, 0.0, self.max_rot_vel)

        self.get_logger().info(f"v_z={v_z:.2f}, u0={u0:.2f}, ω={ω_cmd}")

        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = ω_cmd.tolist()
        self.pub.publish(cmd)

        # store history
        self.last_time  = now
        self.last_z     = self.z
        self.last_phi   = self.phi
        self.last_theta = self.theta
        self.last_psi   = self.psi
        self.last_error_z = self.error_z

def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
