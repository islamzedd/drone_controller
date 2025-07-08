import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PointStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler_scipy(w, x, y, z):
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=False)


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def generate_square_waypoints(size=2.0, height=2.0):
    half = size / 2.0
    return [
        [ half,  half, height],
        [-half,  half, height],
        [-half, -half, height],
        [ half, -half, height],
        [ half,  half, height],
    ]


def generate_figure_eight_waypoints(num_points=200, A=2.0, B=2.0, H=2.0):
    waypoints = []
    t_vals = np.linspace(0, 2*np.pi, num_points)
    for t in t_vals:
        x = A*np.sin(t)
        y = B*np.sin(t)*np.cos(t)
        z = H
        waypoints.append([x, y, z])
    return waypoints


def generate_cool_acrobatic_waypoints(
    num_points=300,
    duration=15.0,
    offset=(5,5,3),
    speed=1.0,
    amp_x1=4.0, amp_x2=2.0,
    amp_y1=3.0, amp_y2=1.5,
    amp_z1=1.8, amp_z2=0.5,
):
    waypoints = []
    t_vals = np.linspace(0, duration, num_points)
    x_off, y_off, z_off = offset
    for t in t_vals:
        t_fast = speed * t
        x = amp_x1*np.sin(2*np.pi*0.3*t_fast) - amp_x2*np.sin(2*np.pi*0.6*t_fast)
        y = amp_y1*np.cos(2*np.pi*0.3*t_fast) - amp_y2*np.cos(2*np.pi*0.6*t_fast)
        z = 2 + amp_z1*np.sin(2*np.pi*0.5*t_fast) + amp_z2*np.sin(2*np.pi*1.2*t_fast)
        waypoints.append([x + x_off, y + y_off, z + z_off])
    return waypoints


class PositionController(Node):
    def __init__(self):
        super().__init__('Position_Controller')
        # Physical parameters
        self.m = 1.0       # mass (kg)
        self.g = 9.81      # gravity (m/s^2)
        self.kt = 1e-5     # thrust coefficient (N/(rad/s)^2)
        self.l = 0.2       # arm length (m)
        self.omega_min = 400.0
        self.omega_max = 800.0

        # Outer-loop PD gains for linearized dynamics
        self.kp_x, self.kd_x = 6.0, 4.0
        self.kp_y, self.kd_y = 6.0, 4.0
        self.kp_z, self.kd_z = 8.0, 6.0

        # Inner attitude PD gains
        self.kp_att_roll, self.kd_att_roll = 6.0, 0.3
        self.kp_att_pitch, self.kd_att_pitch = 6.0, 0.3

        # State
        self.x = self.y = self.z = None
        self.vx = self.vy = self.vz = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.last_x = self.last_y = self.last_z = 0.0
        self.last_time = self.get_clock().now()

        # Trajectory selection
        traj_type = 'figure8'  # 'default', 'figure8', 'acrobatic'
        if traj_type == 'default':
            self.trajectory = generate_square_waypoints(3.0, 4.0)
            self.duration = 20.0
        elif traj_type == 'figure8':
            self.trajectory = generate_figure_eight_waypoints(200, 3.0, 2.0, 4.0)
            self.duration = 20.0
        else:
            self.duration = 15.0
            self.trajectory = generate_cool_acrobatic_waypoints(300, self.duration)
        self.traj_times = np.linspace(0, self.duration, len(self.trajectory))
        self.start_time = None

        # Attitude error storage
        self.e_roll_last = 0.0
        self.e_pitch_last = 0.0

        # ROS topics
        self.sub = self.create_subscription(PoseArray, '/world/quadcopter/pose/info', self.pose_callback, 10)
        self.pub_m = self.create_publisher(Actuators, '/X3/gazebo/command/motor_speed', 10)
        self.pub_ref = self.create_publisher(PointStamped, '/drone/ref_pos', 10)

    def pose_callback(self, msg: PoseArray):
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt < 1e-4:
            return

        # Update state
        x, y, z = msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z
        self.vx = (x - self.last_x)/dt
        self.vy = (y - self.last_y)/dt
        self.vz = (z - self.last_z)/dt
        self.x, self.y, self.z = x, y, z
        self.last_x, self.last_y, self.last_z = x, y, z
        self.last_time = now

        # Orientation
        self.roll, self.pitch, self.yaw = quaternion_to_euler_scipy(
            msg.poses[1].orientation.w,
            msg.poses[1].orientation.x,
            msg.poses[1].orientation.y,
            msg.poses[1].orientation.z)

        self.control_loop(now, dt)

    def control_loop(self, now, dt):
        # Time progression along trajectory
        t = ((now - self.start_time).nanoseconds / 1e9) % self.duration
        idx = np.searchsorted(self.traj_times, t) - 1
        idx = np.clip(idx, 0, len(self.trajectory)-2)
        t1, t2 = self.traj_times[idx], self.traj_times[idx+1]
        wp_prev = np.array(self.trajectory[idx-1]) if idx>0 else np.array(self.trajectory[idx])
        wp_curr = np.array(self.trajectory[idx])
        wp_next = np.array(self.trajectory[idx+1])
        alpha = (t - t1)/(t2 - t1)

        # Desired kinematics
        pos_des = wp_curr*(1-alpha) + wp_next*alpha
        vel_des = (wp_next - wp_curr)/(t2-t1)
        # Second-derivative via central diff
        if idx>0 and idx< len(self.trajectory)-1:
            dt_wp = t2 - t1
            acc_des = (wp_next - 2*wp_curr + wp_prev)/(dt_wp**2)
        else:
            acc_des = np.zeros(3)

        # Errors
        e_pos = pos_des - np.array([self.x, self.y, self.z])
        e_vel = vel_des - np.array([self.vx, self.vy, self.vz])

        # Commanded accelerations
        ax_cmd = acc_des[0] + self.kp_x*e_pos[0] + self.kd_x*e_vel[0]
        ay_cmd = acc_des[1] + self.kp_y*e_pos[1] + self.kd_y*e_vel[1]
        az_cmd = acc_des[2] + self.kp_z*e_pos[2] + self.kd_z*e_vel[2]

        # Total thrust
        T = self.m*(az_cmd + self.g)

        # Invert translational dynamics to Euler angles safely
        denom = az_cmd + self.g
        # Ensure denom not zero
        if abs(denom) < 1e-6:
            self.get_logger().warn('Denominator near zero in attitude calculation, skipping control update')
            return
        # Compute and clamp inputs to domain [-1,1]
        ratio_phi = (-ax_cmd*np.sin(self.yaw) + ay_cmd*np.cos(self.yaw)) / denom
        ratio_theta = ( ax_cmd*np.cos(self.yaw) + ay_cmd*np.sin(self.yaw)) / denom
        ratio_phi = clamp(ratio_phi, -1.0, 1.0)
        ratio_theta = clamp(ratio_theta, -1.0, 1.0)
        # Desired angles, limited to safe tilt
        phi_des = clamp(np.arcsin(ratio_phi), -0.5, 0.5)
        theta_des = clamp(np.arcsin(ratio_theta), -0.5, 0.5)

        # Attitude PD
        e_r = phi_des - self.roll
        de_r = (e_r - self.e_roll_last)/dt
        tau_phi = self.kp_att_roll*e_r + self.kd_att_roll*de_r
        self.e_roll_last = e_r

        e_p = theta_des - self.pitch
        de_p = (e_p - self.e_pitch_last)/dt
        tau_theta = self.kp_att_pitch*e_p + self.kd_att_pitch*de_p
        self.e_pitch_last = e_p

        # Motor mixing (thrust and moments)
        w2 = [0]*4
        w2[0] = (T - tau_phi/self.l + tau_theta/self.l)/(4*self.kt)
        w2[1] = (T + tau_phi/self.l + tau_theta/self.l)/(4*self.kt)
        w2[2] = (T + tau_phi/self.l - tau_theta/self.l)/(4*self.kt)
        w2[3] = (T - tau_phi/self.l - tau_theta/self.l)/(4*self.kt)

        omegas = [clamp(np.sqrt(max(w2_i,0)), self.omega_min, self.omega_max) for w2_i in w2]

        # Publish motor commands
        cmd = Actuators()
        cmd.header = Header(); cmd.header.stamp = now.to_msg()
        cmd.velocity = omegas
        self.pub_m.publish(cmd)

        # Publish reference
        ref = PointStamped(); ref.header.stamp = now.to_msg()
        ref.point.x, ref.point.y, ref.point.z = pos_des
        self.pub_ref.publish(ref)

        self.get_logger().info(f"t={t:.2f}s pos={self.x:.2f},{self.y:.2f},{self.z:.2f} des={pos_des} "
                              f"e_pos={e_pos} e_vel={e_vel}")


def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
