# ROS Python libraries
import rclpy
from rclpy.node import Node
# ROS message libraries
from geometry_msgs.msg import PoseArray, PointStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header

# Basic Python libraries
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler_scipy(w, x, y, z):
    # Reorder to [x, y, z, w] for scipy
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=True)  # Roll, pitch, yaw in degrees

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def generate_square_waypoints(size=2.0, height=2.0):
    """
    Generate waypoints for a square trajectory centered at (0,0) at a fixed height.

    Args:
        size (float): Length of one side of the square (meters)
        height (float): Z-coordinate (altitude) for all waypoints

    Returns:
        List of [x, y, z] waypoints forming a square loop
    """
    half = size / 2.0
    waypoints = [
        [ half,  half, height],  # Top-right
        [-half,  half, height],  # Top-left
        [-half, -half, height],  # Bottom-left
        [ half, -half, height],  # Bottom-right
        [ half,  half, height],  # Back to start to close loop
    ]
    return waypoints

def pid_controller(set_point, actual, last_error, last_integral, dt, kp, ki, kd):
    error = set_point - actual
    integral = last_integral + error * dt
    derivative = (error - last_error) / dt
    return kp * error + ki * integral + kd * derivative, integral, error

class PositionController(Node):
    def __init__(self):
        super().__init__('Position_Controller')

        # Physical parameters
        self.mass = 1.0  # kg (assumed mass of the quadcopter)
        self.g = 9.81    # m/s^2 (gravity)

        # Control gains for feedback linearization
        self.K_p = np.diag([5.0, 5.0, 5.0])  # Position gain
        self.K_d = np.diag([2.5, 2.5, 2.5])  # Velocity gain

        # Memories for PID attitude control
        self.integral_roll = 0.0
        self.last_error_roll = 0.0
        self.integral_pitch = 0.0
        self.last_error_pitch = 0.0
        self.integral_yaw = 0.0
        self.last_error_yaw = 0.0

        # State variables
        self.last_time = self.get_clock().now()
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.current_vx = 0.0  # Velocity estimates (simplified)
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.last_x = None
        self.last_y = None
        self.last_z = None

        # Waypoint tracking
        self.current_wp_idx = 0
        self.wp_threshold = 0.05
        self.trajectory = generate_square_waypoints(size=3.0, height=4.0)

        # ROS subscriptions and publishers
        self.subscription = self.create_subscription(
            PoseArray,
            '/world/quadcopter/pose/info',
            self.pose_callback,
            10)
        self.publisher_ = self.create_publisher(
            Actuators,
            '/X3/gazebo/command/motor_speed',
            10)
        self.ref_pub = self.create_publisher(PointStamped, '/drone/ref_pos', 10)

    def pose_callback(self, msg: PoseArray):
        if len(msg.poses) < 2:
            self.get_logger().warn("PoseArray has fewer than 2 poses")
            return

        # Update current position and orientation
        self.current_x = msg.poses[1].position.x
        self.current_y = msg.poses[1].position.y
        self.current_z = msg.poses[1].position.z
        self.roll, self.pitch, self.yaw = quaternion_to_euler_scipy(
            msg.poses[1].orientation.w,
            msg.poses[1].orientation.x,
            msg.poses[1].position.y,
            msg.poses[1].orientation.z
        )
        self.control_loop()

    def control_loop(self):
        if self.current_x is None or self.last_x is None:
            self.last_x = self.current_x
            self.last_y = self.current_y
            self.last_z = self.current_z
            return

        # Time step
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.001:
            return

        # Estimate velocity (simple differentiation)
        self.current_vx = (self.current_x - self.last_x) / dt
        self.current_vy = (self.current_y - self.last_y) / dt
        self.current_vz = (self.current_z - self.last_z) / dt

        # Update waypoint
        self.desired_pos = self.trajectory[self.current_wp_idx]
        dist = np.linalg.norm([
            self.desired_pos[0] - self.current_x,
            self.desired_pos[1] - self.current_y,
            self.desired_pos[2] - self.current_z
        ])
        if dist < self.wp_threshold:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.trajectory)
            self.desired_pos = self.trajectory[self.current_wp_idx]

        # Feedback linearization control
        e_p = np.array(self.desired_pos) - np.array([self.current_x, self.current_y, self.current_z])
        e_v = np.array([0, 0, 0]) - np.array([self.current_vx, self.current_vy, self.current_vz])  # Desired velocity assumed zero
        u = self.K_p @ e_p + self.K_d @ e_v  # Desired acceleration

        # Gravity compensation
        g_vec = np.array([0, 0, -self.g])
        f_des = self.mass * (u - g_vec)

        # Desired thrust magnitude
        T = np.linalg.norm(f_des)
        if T == 0:
            T = 1e-6  # Avoid division by zero

        # Desired orientation (z-body axis)
        z_b = f_des / T
        desired_yaw = 0.0  # Fixed yaw for simplicity
        x_c = np.array([np.cos(desired_yaw), np.sin(desired_yaw), 0])
        y_b = np.cross(z_b, x_c)
        y_b_norm = np.linalg.norm(y_b)
        if y_b_norm < 1e-6:  # Handle singularity
            y_b = np.array([0, 1, 0])
        else:
            y_b /= y_b_norm
        x_b = np.cross(y_b, z_b)
        R_d = np.column_stack((x_b, y_b, z_b))

        # Convert to Euler angles for attitude control
        r = R.from_matrix(R_d)
        desired_roll, desired_pitch, desired_yaw = r.as_euler('xyz', degrees=True)

        # Attitude control using PID
        omega_roll, self.integral_roll, self.last_error_roll = pid_controller(
            desired_roll, self.roll, self.last_error_roll, self.integral_roll, dt, 1.0, 0.0, 1.0)
        omega_pitch, self.integral_pitch, self.last_error_pitch = pid_controller(
            desired_pitch, self.pitch, self.last_error_pitch, self.integral_pitch, dt, 1.0, 0.0, 1.0)
        omega_yaw, self.integral_yaw, self.last_error_yaw = pid_controller(
            desired_yaw, self.yaw, self.last_error_yaw, self.integral_yaw, dt, 1.0, 0.0, 1.0)

        # Clamp integrals
        self.integral_roll = clamp(self.integral_roll, -1.0, 1.0)
        self.integral_pitch = clamp(self.integral_pitch, -1.0, 1.0)
        self.integral_yaw = clamp(self.integral_yaw, -1.0, 1.0)

        # Map to motor speeds (simplified, adjust based on quadcopter model)
        base_speed = 636.0 + (T / self.mass) * 10.0  # Rough scaling
        motor0 = clamp(base_speed - omega_roll - omega_pitch + omega_yaw, 400.0, 800.0)
        motor1 = clamp(base_speed + omega_roll + omega_pitch + omega_yaw, 400.0, 800.0)
        motor2 = clamp(base_speed + omega_roll - omega_pitch + omega_yaw, 400.0, 800.0)
        motor3 = clamp(base_speed - omega_roll + omega_pitch + omega_yaw, 400.0, 800.0)

        # Publish motor commands
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = [motor0, motor1, motor2, motor3]
        self.publisher_.publish(cmd)

        # Update last states
        self.last_x = self.current_x
        self.last_y = self.current_y
        self.last_z = self.current_z
        self.last_time = now

        # Logging
        self.get_logger().info(
            f"x, y, z: {self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f} m | "
            f"Desired roll, pitch: {desired_roll:.2f}, {desired_pitch:.2f} | "
            f"Command: {motor0:.2f}, {motor1:.2f}, {motor2:.2f}, {motor3:.2f} rad/s"
        )

        # Publish reference position
        ref_msg = PointStamped()
        ref_msg.header.stamp = now.to_msg()
        ref_msg.point.x = float(self.desired_pos[0])
        ref_msg.point.y = float(self.desired_pos[1])
        ref_msg.point.z = float(self.desired_pos[2])
        self.ref_pub.publish(ref_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
