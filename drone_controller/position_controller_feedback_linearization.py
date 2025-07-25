import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header
import numpy as np
import math

class FlHoverController(Node):
    def __init__(self):
        super().__init__('fl_hover_controller')
        # Vehicle parameters
        self.m = 1.5
        self.g = 9.81
        self.I = np.diag([0.0347, 0.0459, 0.0977])
        self.L = 0.244
        self.c_T = 8.54858e-06
        self.c_M = 0.06
        self.max_omega = 1100.0

        # Desired setpoints
        self.declare_parameters('', [
            ('des_x', 2.0), ('des_y', 2.0), ('des_z', 2.0),
            ('des_roll', 0.0), ('des_pitch', 0.0), ('des_yaw', 0.0)
        ])
        # Gains
        self.declare_parameters('', [
            ('kp_pos', 1.2), ('kd_pos', 1.5),
            ('kp_att', 8.0), ('kd_att', 2.5)
        ])

        # Limits
        self.declare_parameters('', [
            ('max_acc', 2.0), ('max_ang_acc', 10.0), ('max_thrust_rate', 2.0)
        ])

        # State
        self.prev_time = self.get_clock().now()
        self.prev_pos = np.zeros(3)
        self.prev_euler = np.zeros(3)
        self.prev_u0 = self.m * self.g

        # Sub/pub
        self.create_subscription(PoseArray, '/world/quadcopter/pose/info', self.callback, 10)
        self.pub = self.create_publisher(Actuators, '/X3/gazebo/command/motor_speed', 10)

    def callback(self, msg: PoseArray):
        if len(msg.poses) < 2:
            return
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds * 1e-9
        if dt < 1e-3:
            return
        dt = min(dt, 0.1)

        # Read state
        p = msg.poses[1].position
        o = msg.poses[1].orientation
        x, y, z = p.x, p.y, p.z
        roll, pitch, yaw = quaternion_to_euler((o.x, o.y, o.z, o.w))

        # Velocities and rates
        vel = (np.array([x, y, z]) - self.prev_pos) / dt
        rates = (np.array([roll, pitch, yaw]) - self.prev_euler) / dt
        self.prev_pos = np.array([x, y, z])
        self.prev_euler = np.array([roll, pitch, yaw])

        # Desired
        des_pos = np.array([
            self.get_parameter('des_x').value,
            self.get_parameter('des_y').value,
            self.get_parameter('des_z').value
        ])
        des_att = np.array([
            self.get_parameter('des_roll').value,
            self.get_parameter('des_pitch').value,
            self.get_parameter('des_yaw').value
        ])

        # Compute errors
        e_pos = des_pos - np.array([x, y, z])
        e_vel = -vel
        e_att = des_att - np.array([roll, pitch, yaw])
        e_rates = -rates

        # Print state and errors
        self.get_logger().info(f"Pos: [{x:.2f}, {y:.2f}, {z:.2f}] Vel: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
        self.get_logger().info(f"Errors -> Pos: {e_pos}, Vel: {e_vel}, Att: {e_att}, Rates: {e_rates}")

        # Feedback linearization for translational dynamics
        kp_pos = self.get_parameter('kp_pos').value
        kd_pos = self.get_parameter('kd_pos').value
        a_des = kp_pos * e_pos + kd_pos * e_vel
        f = self.m * (a_des + np.array([0, 0, self.g]))

        # Rotation matrix
        cr = math.cos(roll); sr = math.sin(roll)
        cp = math.cos(pitch); sp = math.sin(pitch)
        cy = math.cos(yaw); sy = math.sin(yaw)
        R = np.array([
            [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
            [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
            [-sp,   sr*cp,           cr*cp]
        ])

        # Thrust u0
        u = R.T.dot(f)
        u0 = np.clip(u[2], 0.5*self.m*self.g, 1.5*self.m*self.g)
        drip = self.get_parameter('max_thrust_rate').value
        u0 = np.clip(u0, self.prev_u0 - drip*dt, self.prev_u0 + drip*dt)
        self.prev_u0 = u0

        # Feedback linearization for rotational dynamics
        kp_att = self.get_parameter('kp_att').value
        kd_att = self.get_parameter('kd_att').value
        alpha_des = kp_att * e_att + kd_att * e_rates
        omega_body = np.array([roll, pitch, yaw])
        tau = self.I.dot(alpha_des) + np.cross(omega_body, self.I.dot(rates))

        # Print control targets
        self.get_logger().info(f"Desired a: {a_des}, u0: {u0}, alpha_des: {alpha_des}, tau: {tau}")

        # Mixer and motor speeds
        b = np.hstack((u0, tau))
        mix = np.array([
            [1, 1, 1, 1],
            [0, self.L, 0, -self.L],
            [-self.L, 0, self.L, 0],
            [self.c_M, -self.c_M, self.c_M, -self.c_M]
        ])
        thrusts = np.linalg.solve(mix, b)
        omega = np.sqrt(np.clip(thrusts/self.c_T, 0, None))
        omega = np.clip(omega, 0, self.max_omega)

        # Print motor commands
        self.get_logger().info(f"Motor omegas: {omega}")

        # Publish
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = omega.tolist()
        self.pub.publish(cmd)
        self.prev_time = now


def quaternion_to_euler(q):
    x, y, z, w = q
    t0 = 2.0 * (w*x + y*z)
    t1 = 1.0 - 2.0*(x*x + y*y)
    roll  = math.atan2(t0, t1)
    t2 = 2.0*(w*y - z*x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw   = math.atan2(t3, t4)
    return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = FlHoverController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
