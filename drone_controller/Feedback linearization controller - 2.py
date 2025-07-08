# ROS PY libs
import rclpy
from rclpy.node import Node
# ROS msg Libs
from geometry_msgs.msg import PoseArray, PointStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header

# Basic python libs
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler_scipy(w, x, y, z):
    # Reorder to [x, y, z, w] for scipy
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=True)  # Roll, pitch, yaw in degree

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def generate_square_waypoints(size=2.0, height=2.0):
    """
    Generate waypoints for a square trajectory centered at (0,0) at a fixed height.

    Args:
        size (float): length of one side of the square (meters)
        height (float): z-coordinate (altitude) for all waypoints

    Returns:
        list of [x, y, z] waypoints forming a square loop
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

def generate_figure_eight_waypoints(num_points=7, A=2.0, B=2.0, H=2.0):
    waypoints = []
    t_vals = np.linspace(0, 2 * np.pi, num_points)
    for t in t_vals:
        x = A * np.sin(t)
        y = B * np.sin(t) * np.cos(t)
        z = H
        waypoints.append([x, y, z])
    return waypoints


def generate_cool_acrobatic_waypoints(
    num_points=300, 
    duration=15.0, 
    offset=(5, 5, 3),
    speed=1.0,
    amp_x1=4.0, amp_x2=2.0,
    amp_y1=3.0, amp_y2=1.5,
    amp_z1=1.8, amp_z2=0.5,
):
    waypoints = []
    t_vals = np.linspace(0, duration, num_points)
    x_off, y_off, z_off = offset

    for t in t_vals:
        t_fast = speed * t  # scale time for faster/slower trajectory

        # Twisting figure-8 style on x-y with amplitude parameters
        x = amp_x1 * np.sin(2 * np.pi * 0.3 * t_fast) - amp_x2 * np.sin(2 * np.pi * 0.6 * t_fast)
        y = amp_y1 * np.cos(2 * np.pi * 0.3 * t_fast) - amp_y2 * np.cos(2 * np.pi * 0.6 * t_fast)

        # Vertical oscillation with amplitude parameters
        z = 2 + amp_z1 * np.sin(2 * np.pi * 0.5 * t_fast) + amp_z2 * np.sin(2 * np.pi * 1.2 * t_fast)

        # Add offsets
        x += x_off
        y += y_off
        z += z_off

        waypoints.append([x, y, z])

    return waypoints

class PositionController(Node):
    def __init__(self):
        super().__init__('Position_Controller')

        # Drone physical parameters
        self.m = 1.5  # Mass in kg
        self.g = 9.81  # Gravity in m/s^2
        
        # State variables
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.prev_position = np.zeros(3)
        self.prev_velocity = np.zeros(3)
        
        # Orientation variables
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Tracking variables
        self.current_wp_idx = 0
        self.wp_threshold = 0.05
        
        # Control gains for feedback linearization
        self.kp = np.array([8.0, 8.0, 10.0])  # Position gains
        self.kd = np.array([4.0, 4.0, 6.0])   # Velocity gains
        
        # Attitude control gains
        self.kp_att = np.array([2.0, 2.0])    # Roll, pitch gains
        self.kd_att = np.array([0.5, 0.5])    # Roll, pitch derivative gains
        
        # Integral terms for attitude control
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.last_error_roll = 0.0
        self.last_error_pitch = 0.0
        
        # Time management
        self.last_time = self.get_clock().now()
        self.first_callback = True

        # Trajectory generation
        traj_type = "default"  # "default", "figure8", "acrobatic"

        if traj_type == "default":
            self.trajectory = generate_square_waypoints(size=3.0, height=4.0)
        elif traj_type == "figure8":
            self.trajectory = generate_figure_eight_waypoints(num_points=20, A=3.0, B=2.0, H=5.0)
        elif traj_type == "acrobatic":
            self.trajectory = generate_cool_acrobatic_waypoints(num_points=200, duration=20)

        # ROS subscribers and publishers
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
        
        # Update position
        self.prev_position = self.position.copy()
        self.position[0] = msg.poses[1].position.x
        self.position[1] = msg.poses[1].position.y
        self.position[2] = msg.poses[1].position.z
        
        # Update orientation
        self.roll, self.pitch, self.yaw = quaternion_to_euler_scipy(
            msg.poses[1].orientation.w,
            msg.poses[1].orientation.x,
            msg.poses[1].orientation.y,
            msg.poses[1].orientation.z
        )
        
        # Convert degrees to radians for calculations
        self.roll_rad = np.radians(self.roll)
        self.pitch_rad = np.radians(self.pitch)
        self.yaw_rad = np.radians(self.yaw)
        
        self.control_loop()

    def control_loop(self):
        if self.first_callback:
            self.first_callback = False
            return

        # Time calculation
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.001:
            return

        # Calculate velocity using finite difference
        self.prev_velocity = self.velocity.copy()
        self.velocity = (self.position - self.prev_position) / dt
        
        # Waypoint management
        if self.current_wp_idx >= len(self.trajectory):
            self.current_wp_idx = 0

        desired_pos = np.array(self.trajectory[self.current_wp_idx])
        
        # Check if we've reached the current waypoint
        dist = np.linalg.norm(desired_pos - self.position)
        if dist < self.wp_threshold:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.trajectory):
                self.current_wp_idx = 0
            desired_pos = np.array(self.trajectory[self.current_wp_idx])

        # Desired velocity and acceleration (for smoother tracking)
        desired_vel = np.zeros(3)  # Could be computed from trajectory derivative
        desired_acc = np.zeros(3)  # Could be computed from trajectory second derivative
        
        # Feedback linearization control law
        position_error = desired_pos - self.position
        velocity_error = desired_vel - self.velocity
        
        # Compute desired acceleration using feedback linearization
        u = desired_acc + self.kp * position_error + self.kd * velocity_error
        
        # Extract desired thrust and attitude commands
        u_total = np.linalg.norm(u + np.array([0, 0, self.g]))
        thrust = self.m * u_total
        
        # Compute desired roll and pitch angles
        if u_total > 0.1:  # Avoid division by zero
            desired_roll_rad = np.arcsin(clamp(
                (u[1] * np.cos(self.yaw_rad) - u[0] * np.sin(self.yaw_rad)) / u_total,
                -0.3, 0.3))  # Limit to ±17 degrees
            desired_pitch_rad = np.arcsin(clamp(
                (u[0] * np.cos(self.yaw_rad) + u[1] * np.sin(self.yaw_rad)) / u_total,
                -0.3, 0.3))  # Limit to ±17 degrees
        else:
            desired_roll_rad = 0.0
            desired_pitch_rad = 0.0
        
        # Attitude control using PD controller
        roll_error = desired_roll_rad - self.roll_rad
        pitch_error = desired_pitch_rad - self.pitch_rad
        
        # Angular velocities for attitude control
        omega_roll = self.kp_att[0] * roll_error + self.kd_att[0] * (roll_error - self.last_error_roll) / dt
        omega_pitch = self.kp_att[1] * pitch_error + self.kd_att[1] * (pitch_error - self.last_error_pitch) / dt
        
        # Update last errors
        self.last_error_roll = roll_error
        self.last_error_pitch = pitch_error
        
        # Convert thrust to omega_z (assuming linear relationship)
        base_thrust = self.m * self.g  # Hover thrust
        omega_z = (thrust - base_thrust) * 0.5  # Scaling factor
        
        # Clamp control outputs
        omega_roll = clamp(omega_roll, -50, 50)
        omega_pitch = clamp(omega_pitch, -50, 50)
        omega_z = clamp(omega_z, -100, 100)
        
        # Motor mixing
        motor0 = clamp(636.0 - omega_roll - omega_pitch + omega_z, 400.0, 800.0)
        motor1 = clamp(636.0 + omega_roll + omega_pitch + omega_z, 400.0, 800.0)
        motor2 = clamp(636.0 + omega_roll - omega_pitch + omega_z, 400.0, 800.0)
        motor3 = clamp(636.0 - omega_roll + omega_pitch + omega_z, 400.0, 800.0)

        # Publish motor commands
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = [motor0, motor1, motor2, motor3]
        self.publisher_.publish(cmd)

        # Update time
        self.last_time = now

        # Logging
        self.get_logger().info(f"Pos: [{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}] m | "
                             f"Desired: [{desired_pos[0]:.2f}, {desired_pos[1]:.2f}, {desired_pos[2]:.2f}] m | "
                             f"Error: [{position_error[0]:.2f}, {position_error[1]:.2f}, {position_error[2]:.2f}] m | "
                             f"Motors: [{motor0:.1f}, {motor1:.1f}, {motor2:.1f}, {motor3:.1f}] rad/s")

        # Publish reference position for plotting
        ref_msg = PointStamped()
        ref_msg.header.stamp = now.to_msg()
        ref_msg.point.x = float(desired_pos[0])
        ref_msg.point.y = float(desired_pos[1])
        ref_msg.point.z = float(desired_pos[2])
        self.ref_pub.publish(ref_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()