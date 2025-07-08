import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PointStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header

import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler_scipy(w, x, y, z):
    # Convert quaternion to roll, pitch, yaw (degrees)
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=True)


def clamp(val, mn, mx):
    return max(min(val, mx), mn)


def pid_controller(set_point, actual, last_err, last_int, dt, kp, ki, kd):
    err = set_point - actual
    integral = last_int + err * dt
    derivative = (err - last_err) / dt if dt > 0 else 0.0
    out = kp * err + ki * integral + kd * derivative
    return out, integral, err


def generate_square_waypoints(size=4.0, height=2.0):
    half = size / 2.0
    return [
        [ half,  half, height],
        [-half,  half, height],
        [-half, -half, height],
        [ half, -half, height],
        [ half,  half, height],
    ]


class FeedbackLinearizedPositionController(Node):
    def __init__(self):
        super().__init__('fb_lin_square_controller')

        # Vehicle parameters from Gazebo Iris SDF
        self.m   = 1.5       # mass [kg] from <mass>1.5</mass>
        self.Ixx = 0.0347563 # inertia [kg·m²] from <ixx>0.0347563</ixx>
        self.Iyy = 0.0458929 # inertia [kg·m²] from <iyy>0.0458929</iyy>
        self.Izz = 0.0977    # inertia [kg·m²] from <izz>0.0977</izz>
        self.b   = 5.84e-06  # thrust coefficient (motorConstant)
        self.d   = 0.06      # moment coefficient (momentConstant)
        self.L   = 0.3       # arm length [m]

        # PID gains for position → v_x,v_y,v_z (initial tuning values)
        self.Kp_x, self.Ki_x, self.Kd_x = 1.5, 0.0, 0.5
        self.Kp_y, self.Ki_y, self.Kd_y = 1.5, 0.0, 0.5
        self.Kp_z, self.Ki_z, self.Kd_z = 30.0, 5.0, 40.0

        # PID gains for attitude → v_phi,v_theta,v_psi (initial tuning values)
        self.Kp_phi,   self.Ki_phi,   self.Kd_phi   = 8.0, 0.2, 2.0
        self.Kp_theta, self.Ki_theta, self.Kd_theta = 8.0, 0.2, 2.0
        self.Kp_psi,   self.Ki_psi,   self.Kd_psi   = 4.0, 0.1, 1.0

        # PID state
        self.err_x = self.err_y = self.err_z = 0.0
        self.int_x = self.int_y = self.int_z = 0.0
        self.err_phi = self.err_theta = self.err_psi = 0.0
        self.int_phi = self.int_theta = self.int_psi = 0.0

        # Pose and orientation
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0

        # Square trajectory
        self.trajectory = generate_square_waypoints(size=4.0, height=2.0)
        self.wp_idx = 0
        self.wp_thresh = 0.1

        self.last_time = self.get_clock().now()

        # Subscribers and publishers
        self.sub = self.create_subscription(PoseArray, '/world/quadcopter/pose/info', self.pose_cb, 10)
        self.pub = self.create_publisher(Actuators, '/X3/gazebo/command/motor_speed', 10)
        self.ref_pub = self.create_publisher(PointStamped, '/drone/ref_pos', 10)

    def pose_cb(self, msg: PoseArray):
        if len(msg.poses) < 2:
            return
        p = msg.poses[1].position
        self.x, self.y, self.z = p.x, p.y, p.z
        self.roll, self.pitch, self.yaw = quaternion_to_euler_scipy(
            msg.poses[1].orientation.w,
            msg.poses[1].orientation.x,
            msg.poses[1].orientation.y,
            msg.poses[1].orientation.z
        )
        self.control_loop()

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt < 1e-3:
            return

        # Waypoint update
        wp = self.trajectory[self.wp_idx]
        dist = np.linalg.norm([wp[0]-self.x, wp[1]-self.y, wp[2]-self.z])
        if dist < self.wp_thresh:
            self.wp_idx = (self.wp_idx + 1) % len(self.trajectory)
            wp = self.trajectory[self.wp_idx]

        # 1) Position PID → v_x,v_y,v_z
        v_x, self.int_x, self.err_x = pid_controller(wp[0], self.x, self.err_x, self.int_x, dt,
                                                     self.Kp_x, self.Ki_x, self.Kd_x)
        v_y, self.int_y, self.err_y = pid_controller(wp[1], self.y, self.err_y, self.int_y, dt,
                                                     self.Kp_y, self.Ki_y, self.Kd_y)
        v_z, self.int_z, self.err_z = pid_controller(wp[2], self.z, self.err_z, self.int_z, dt,
                                                     self.Kp_z, self.Ki_z, self.Kd_z)

        # 2) Compute u0
        g = 9.81
        a_d = np.array([v_x, v_y, v_z + g])
        norm_ad = np.linalg.norm(a_d)
        u0 = self.m * norm_ad

        # 3) Invert for phi_d, theta_d, psi_d
        theta_d = np.arcsin((v_x*np.cos(self.yaw) + v_y*np.sin(self.yaw)) / norm_ad)
        phi_d   = np.arctan2((v_x*np.sin(self.yaw) - v_y*np.cos(self.yaw)), (v_z + g))
        psi_d   = self.yaw  # hold yaw

        # 4) Attitude PID → v_phi,v_theta,v_psi
        v_phi,   self.int_phi,   self.err_phi   = pid_controller(phi_d,   self.roll,  self.err_phi,   self.int_phi,   dt,
                                                                 self.Kp_phi, self.Ki_phi, self.Kd_phi)
        v_theta, self.int_theta, self.err_theta = pid_controller(theta_d, self.pitch, self.err_theta, self.int_theta, dt,
                                                                 self.Kp_theta, self.Ki_theta, self.Kd_theta)
        v_psi,   self.int_psi,   self.err_psi   = pid_controller(psi_d,   self.yaw,   self.err_psi,   self.int_psi,   dt,
                                                                 self.Kp_psi, self.Ki_psi, self.Kd_psi)

        # 5) Cancellation → u1,u2,u3
        dot_phi = dot_theta = dot_psi = 0.0  # or obtain from sensors
        u1 = self.Ixx*v_phi   - (self.Iyy - self.Izz)*dot_theta*dot_psi
        u2 = self.Iyy*v_theta - (self.Izz - self.Ixx)*dot_phi*dot_psi
        u3 = self.Izz*v_psi   - (self.Ixx - self.Iyy)*dot_phi*dot_theta

        # 6) Motor mixing
        b, L, d = self.b, self.L, self.d
        M = np.array([[ b,    b,    b,    b   ],
                      [ 0,    b*L,  0,   -b*L ],
                      [-b*L,  0,    b*L,  0   ],
                      [ d,   -d,    d,   -d   ]])
        u_vec = np.array([u0, u1, u2, u3])
        omega2 = np.linalg.solve(M, u_vec)
        omegas = np.sqrt(np.clip(omega2, 0, None))
        omegas = [clamp(o, 400.0, 800.0) for o in omegas]

        # Publish actuator commands
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = omegas
        self.pub.publish(cmd)

        # Publish reference waypoint
        ref = PointStamped()
        ref.header = Header()
        ref.header.stamp = now.to_msg()
        ref.point.x, ref.point.y, ref.point.z = wp
        self.ref_pub.publish(ref)

        self.last_time = now


def main(args=None):
    rclpy.init(args=args)
    node = FeedbackLinearizedPositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
