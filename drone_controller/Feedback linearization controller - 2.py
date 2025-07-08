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

class FeedbackLinearizationController(Node):
    def __init__(self):
        super().__init__('Feedback_Linearization_Controller')

        # Quadcopter physical parameters
        self.mass = 1.3  # kg
        self.gravity = 9.81  # m/s^2
        self.Ixx = 0.029  # kg*m^2
        self.Iyy = 0.029  # kg*m^2
        self.Izz = 0.055  # kg*m^2
        self.arm_length = 0.33  # m
        self.thrust_coefficient = 8.54e-06  # N*s^2/rad^2
        self.drag_coefficient = 0.016  # N*m*s^2/rad^2

        # State variables
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.angular_velocity_x = 0.0
        self.angular_velocity_y = 0.0
        self.angular_velocity_z = 0.0
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None

        # Trajectory following
        self.current_wp_idx = 0
        self.wp_threshold = 0.05
        self.trajectory = generate_square_waypoints(size=3.0, height=4.0)

        # Feedback linearization gains (reduced for stability)
        # Position control gains (outer loop)
        self.kp_pos = np.array([2.0, 2.0, 5.0])  # [x, y, z]
        self.kd_pos = np.array([1.5, 1.5, 3.0])   # [x, y, z]
        
        # Attitude control gains (inner loop)
        self.kp_att = np.array([4.0, 4.0, 2.0])   # [roll, pitch, yaw]
        self.kd_att = np.array([1.0, 1.0, 0.5])   # [roll, pitch, yaw]

        # Time tracking
        self.last_time = self.get_clock().now()
        self.dt = 0.01  # Default dt

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
        
        # Store previous values for velocity calculation
        self.previous_x = self.current_x
        self.previous_y = self.current_y
        self.previous_z = self.current_z
        self.previous_roll = self.roll
        self.previous_pitch = self.pitch
        self.previous_yaw = self.yaw
        
        # Update current position and orientation
        self.current_x = msg.poses[1].position.x
        self.current_y = msg.poses[1].position.y
        self.current_z = msg.poses[1].position.z
        
        self.roll, self.pitch, self.yaw = quaternion_to_euler_scipy(
            msg.poses[1].orientation.w,
            msg.poses[1].orientation.x,
            msg.poses[1].orientation.y,
            msg.poses[1].orientation.z
        )
        
        # Convert angles to radians
        self.roll = np.radians(self.roll)
        self.pitch = np.radians(self.pitch)
        self.yaw = np.radians(self.yaw)
        
        # Calculate time step
        now = self.get_clock().now()
        self.dt = (now - self.last_time).nanoseconds / 1e9
        if self.dt <= 0.001:
            return
        
        # Calculate velocities (numerical differentiation with filtering)
        if self.previous_x is not None:
            # Simple low-pass filter for velocity estimation
            alpha = 0.3  # Filter coefficient
            raw_vx = (self.current_x - self.previous_x) / self.dt
            raw_vy = (self.current_y - self.previous_y) / self.dt
            raw_vz = (self.current_z - self.previous_z) / self.dt
            
            self.velocity_x = alpha * raw_vx + (1 - alpha) * self.velocity_x
            self.velocity_y = alpha * raw_vy + (1 - alpha) * self.velocity_y
            self.velocity_z = alpha * raw_vz + (1 - alpha) * self.velocity_z
            
            # Angular velocities with filtering
            raw_wx = (self.roll - self.previous_roll) / self.dt
            raw_wy = (self.pitch - self.previous_pitch) / self.dt
            raw_wz = (self.yaw - self.previous_yaw) / self.dt
            
            self.angular_velocity_x = alpha * raw_wx + (1 - alpha) * self.angular_velocity_x
            self.angular_velocity_y = alpha * raw_wy + (1 - alpha) * self.angular_velocity_y
            self.angular_velocity_z = alpha * raw_wz + (1 - alpha) * self.angular_velocity_z
        
        self.last_time = now
        self.control_loop()

    def control_loop(self):
        if self.current_z is None:
            return

        # Waypoint management
        if self.current_wp_idx >= len(self.trajectory):
            self.current_wp_idx = 0

        desired_pos = self.trajectory[self.current_wp_idx]
        
        # Check if we've reached the current waypoint
        dist = np.linalg.norm([
            desired_pos[0] - self.current_x,
            desired_pos[1] - self.current_y,
            desired_pos[2] - self.current_z
        ])

        if dist < self.wp_threshold:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.trajectory):
                self.current_wp_idx = 0

        desired_pos = self.trajectory[self.current_wp_idx]

        # Feedback linearization control
        motor_speeds = self.feedback_linearization_control(desired_pos)
        
        # Publish motor commands
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.velocity = motor_speeds
        self.publisher_.publish(cmd)

        # Publish reference position for plotting
        ref_msg = PointStamped()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.point.x = desired_pos[0]
        ref_msg.point.y = desired_pos[1]
        ref_msg.point.z = desired_pos[2]
        self.ref_pub.publish(ref_msg)

        self.get_logger().info(f"Pos: [{self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}] | "
                              f"Desired: [{desired_pos[0]:.2f}, {desired_pos[1]:.2f}, {desired_pos[2]:.2f}] | "
                              f"Motors: [{motor_speeds[0]:.0f}, {motor_speeds[1]:.0f}, {motor_speeds[2]:.0f}, {motor_speeds[3]:.0f}]")

    def feedback_linearization_control(self, desired_pos):
        """
        Feedback linearization controller for quadcopter
        """
        # Current state
        current_state = np.array([self.current_x, self.current_y, self.current_z])
        current_velocity = np.array([self.velocity_x, self.velocity_y, self.velocity_z])
        current_attitude = np.array([self.roll, self.pitch, self.yaw])
        current_angular_velocity = np.array([self.angular_velocity_x, self.angular_velocity_y, self.angular_velocity_z])
        
        # Desired state (assuming zero desired velocity and acceleration for waypoint tracking)
        desired_position = np.array(desired_pos)
        desired_velocity = np.array([0.0, 0.0, 0.0])
        desired_acceleration = np.array([0.0, 0.0, 0.0])
        
        # Position errors
        position_error = desired_position - current_state
        velocity_error = desired_velocity - current_velocity
        
        # Virtual control inputs for position (desired accelerations)
        virtual_control = desired_acceleration + self.kp_pos * position_error + self.kd_pos * velocity_error
        
        # Desired total thrust (along body z-axis) - corrected formulation
        # Add gravity compensation and limit thrust command
        thrust_command = virtual_control[2] + self.gravity
        desired_thrust = self.mass * thrust_command
        
        # Limit thrust to reasonable bounds
        max_thrust = 4 * self.thrust_coefficient * (750.0**2)  # 4 motors at 750 rad/s
        min_thrust = 4 * self.thrust_coefficient * (450.0**2)  # 4 motors at 450 rad/s
        desired_thrust = clamp(desired_thrust, min_thrust, max_thrust)
        
        # Desired attitude angles from virtual control (corrected signs)
        if abs(thrust_command) > 0.1:  # Avoid division by very small numbers
            desired_roll = -(virtual_control[1] * np.cos(self.yaw) - virtual_control[0] * np.sin(self.yaw)) / thrust_command
            desired_pitch = (virtual_control[0] * np.cos(self.yaw) + virtual_control[1] * np.sin(self.yaw)) / thrust_command
        else:
            desired_roll = 0.0
            desired_pitch = 0.0
        
        desired_yaw = 0.0  # Keep yaw at 0 for simplicity
        
        # Clamp desired angles to reasonable limits (±20 degrees)
        desired_roll = clamp(desired_roll, -0.35, 0.35)
        desired_pitch = clamp(desired_pitch, -0.35, 0.35)
        
        # Attitude errors
        desired_attitude = np.array([desired_roll, desired_pitch, desired_yaw])
        desired_angular_velocity = np.array([0.0, 0.0, 0.0])  # Assuming zero desired angular velocity
        
        attitude_error = desired_attitude - current_attitude
        angular_velocity_error = desired_angular_velocity - current_angular_velocity
        
        # Virtual control inputs for attitude (desired angular accelerations)
        virtual_torque = self.kp_att * attitude_error + self.kd_att * angular_velocity_error
        
        # Convert virtual inputs to actual control inputs
        thrust = desired_thrust
        tau_x = virtual_torque[0] * self.Ixx
        tau_y = virtual_torque[1] * self.Iyy
        tau_z = virtual_torque[2] * self.Izz
        
        # Convert thrust and torques to motor speeds
        motor_speeds = self.control_allocation(thrust, tau_x, tau_y, tau_z)
        
        return motor_speeds

    def control_allocation(self, thrust, tau_x, tau_y, tau_z):
        """
        Convert thrust and torques to individual motor speeds
        Improved control allocation for X-configuration quadcopter
        """
        # Control allocation matrix for X-configuration quadcopter
        # Motor arrangement (standard X-config):
        # 0: Front-right, 1: Back-left, 2: Front-left, 3: Back-right
        
        # Mixer matrix for X-configuration
        # Each motor contributes to thrust and torques
        l = self.arm_length / np.sqrt(2)  # Effective arm length for X-config
        
        # Thrust force from each motor
        f0 = thrust/4 - tau_x/(4*l) - tau_y/(4*l) - tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        f1 = thrust/4 + tau_x/(4*l) + tau_y/(4*l) - tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        f2 = thrust/4 + tau_x/(4*l) - tau_y/(4*l) + tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        f3 = thrust/4 - tau_x/(4*l) + tau_y/(4*l) + tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        
        # Convert forces to motor speeds (omega = sqrt(F / k_thrust))
        motor_speeds = []
        forces = [f0, f1, f2, f3]
        
        for force in forces:
            if force > 0:
                omega = np.sqrt(abs(force) / self.thrust_coefficient)
                motor_speed = clamp(omega, 400.0, 800.0)
            else:
                motor_speed = 400.0  # Minimum motor speed
            motor_speeds.append(motor_speed)
        
        return motor_speeds


def main(args=None):
    rclpy.init(args=args)
    node = FeedbackLinearizationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
