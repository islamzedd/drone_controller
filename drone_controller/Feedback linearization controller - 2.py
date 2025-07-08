# ROS PY libs
import rclpy
from rclpy.node import Node
# ROS msg Libs
from geometry_msgs.msg import PoseArray, PointStamped, TwistStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header

# Basic python libs
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler_scipy(w, x, y, z):
    # Reorder to [x, y, z, w] for scipy
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=True)  # Roll, pitch, yaw in degrees

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert euler angles to rotation matrix"""
    # Convert to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    return R_z @ R_y @ R_x

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def generate_square_waypoints(size=2.0, height=2.0):
    """
    Generate waypoints for a square trajectory centered at (0,0) at a fixed height.
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
        self.mass = 1.5  # kg
        self.gravity = 9.81  # m/s^2
        self.Ixx = 0.029  # kg*m^2
        self.Iyy = 0.029  # kg*m^2
        self.Izz = 0.055  # kg*m^2
        self.arm_length = 0.23  # m
        self.thrust_coefficient = 8.54e-6  # N*s^2/rad^2
        self.drag_coefficient = 0.016  # N*m*s^2/rad^2
        
        # State variables
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.euler_angles = np.zeros(3)  # [roll, pitch, yaw] in degrees
        self.angular_velocity = np.zeros(3)  # [p, q, r] in rad/s
        
        # Control gains for outer loop (position control)
        self.kp_pos = np.array([8.0, 8.0, 10.0])  # Position gains
        self.kd_pos = np.array([4.0, 4.0, 6.0])   # Velocity gains
        
        # Control gains for inner loop (attitude control)
        self.kp_att = np.array([6.0, 6.0, 4.0])   # Attitude gains
        self.kd_att = np.array([2.0, 2.0, 1.0])   # Angular velocity gains
        
        # Trajectory following
        self.trajectory = generate_square_waypoints(size=4.0, height=3.0)
        self.current_wp_idx = 0
        self.wp_threshold = 0.3
        
        # Desired trajectory derivatives (for feedforward)
        self.desired_position = np.zeros(3)
        self.desired_velocity = np.zeros(3)
        self.desired_acceleration = np.zeros(3)
        
        # Time management
        self.last_time = self.get_clock().now()
        self.dt = 0.01  # Expected control loop time
        
        # ROS subscribers and publishers
        self.subscription = self.create_subscription(
            PoseArray,
            '/world/quadcopter/pose/info',
            self.pose_callback,
            10)
        
        self.velocity_subscription = self.create_subscription(
            TwistStamped,
            '/world/quadcopter/velocity/info',
            self.velocity_callback,
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
        
        # Extract position and orientation
        self.position[0] = msg.poses[1].position.x
        self.position[1] = msg.poses[1].position.y
        self.position[2] = msg.poses[1].position.z
        
        roll, pitch, yaw = quaternion_to_euler_scipy(
            msg.poses[1].orientation.w,
            msg.poses[1].orientation.x,
            msg.poses[1].orientation.y,
            msg.poses[1].orientation.z
        )
        self.euler_angles = np.array([roll, pitch, yaw])
        
        self.control_loop()

    def velocity_callback(self, msg: TwistStamped):
        """Get velocity feedback (if available)"""
        self.velocity[0] = msg.twist.linear.x
        self.velocity[1] = msg.twist.linear.y
        self.velocity[2] = msg.twist.linear.z
        
        self.angular_velocity[0] = msg.twist.angular.x
        self.angular_velocity[1] = msg.twist.angular.y
        self.angular_velocity[2] = msg.twist.angular.z

    def update_trajectory(self):
        """Update current waypoint and desired trajectory"""
        if self.current_wp_idx >= len(self.trajectory):
            self.current_wp_idx = 0
        
        current_target = self.trajectory[self.current_wp_idx]
        self.desired_position = np.array(current_target)
        
        # Calculate distance to current waypoint
        dist = np.linalg.norm(self.desired_position - self.position)
        
        if dist < self.wp_threshold:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.trajectory):
                self.current_wp_idx = 0
        
        # For simplicity, set desired velocity and acceleration to zero
        # In practice, you would compute these from trajectory derivatives
        self.desired_velocity = np.zeros(3)
        self.desired_acceleration = np.zeros(3)

    def feedback_linearization_control(self):
        """
        Feedback linearization controller for quadcopter
        """
        # Update trajectory
        self.update_trajectory()
        
        # Position errors
        position_error = self.desired_position - self.position
        velocity_error = self.desired_velocity - self.velocity
        
        # Outer loop: Position control to desired acceleration
        desired_acc = (self.desired_acceleration + 
                      self.kp_pos * position_error + 
                      self.kd_pos * velocity_error)
        
        # Convert euler angles to radians for calculations
        roll_rad = np.radians(self.euler_angles[0])
        pitch_rad = np.radians(self.euler_angles[1])
        yaw_rad = np.radians(self.euler_angles[2])
        
        # Rotation matrix from body to world frame
        R = euler_to_rotation_matrix(self.euler_angles[0], 
                                   self.euler_angles[1], 
                                   self.euler_angles[2])
        
        # Desired total thrust (along z-axis in world frame)
        desired_acc_world = desired_acc + np.array([0, 0, self.gravity])
        
        # Total thrust magnitude
        thrust_magnitude = np.linalg.norm(desired_acc_world) * self.mass
        
        # Desired thrust direction (unit vector)
        if thrust_magnitude > 0:
            thrust_direction = desired_acc_world / np.linalg.norm(desired_acc_world)
        else:
            thrust_direction = np.array([0, 0, 1])
        
        # Desired attitude from thrust direction
        # For small angles approximation:
        desired_roll = np.arcsin(-thrust_direction[1])
        desired_pitch = np.arcsin(thrust_direction[0] / np.cos(desired_roll))
        desired_yaw = np.radians(self.euler_angles[2])  # Keep current yaw
        
        # Convert to degrees
        desired_attitude = np.array([np.degrees(desired_roll), 
                                   np.degrees(desired_pitch), 
                                   np.degrees(desired_yaw)])
        
        # Attitude errors
        attitude_error = desired_attitude - self.euler_angles
        
        # Wrap yaw error to [-180, 180]
        attitude_error[2] = ((attitude_error[2] + 180) % 360) - 180
        
        # Desired angular velocities (set to zero for simplicity)
        desired_angular_velocity = np.zeros(3)
        angular_velocity_error = desired_angular_velocity - np.degrees(self.angular_velocity)
        
        # Inner loop: Attitude control to desired angular acceleration
        desired_angular_acc = (self.kp_att * attitude_error + 
                             self.kd_att * angular_velocity_error)
        
        # Convert to torques using inertia matrix
        I = np.diag([self.Ixx, self.Iyy, self.Izz])
        desired_torques = I @ np.radians(desired_angular_acc)
        
        # Convert thrust and torques to motor commands
        motor_speeds = self.allocation_matrix(thrust_magnitude, desired_torques)
        
        return motor_speeds

    def allocation_matrix(self, thrust, torques):
        """
        Convert thrust and torques to motor speeds using allocation matrix
        """
        # Allocation matrix for X-configuration quadcopter
        # motor layout: 
        #   1     0
        #     \ /
        #      X
        #     / \
        #   2     3
        
        # Thrust contribution per motor
        thrust_per_motor = thrust / 4.0
        
        # Torque distribution
        tau_x = torques[0]  # Roll torque
        tau_y = torques[1]  # Pitch torque
        tau_z = torques[2]  # Yaw torque
        
        # Motor thrust commands (N)
        T0 = thrust_per_motor - tau_x/(2*self.arm_length) - tau_y/(2*self.arm_length) - tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        T1 = thrust_per_motor + tau_x/(2*self.arm_length) + tau_y/(2*self.arm_length) - tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        T2 = thrust_per_motor + tau_x/(2*self.arm_length) - tau_y/(2*self.arm_length) + tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        T3 = thrust_per_motor - tau_x/(2*self.arm_length) + tau_y/(2*self.arm_length) + tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        
        # Convert thrust to motor speeds (rad/s)
        # T = k_t * omega^2, so omega = sqrt(T/k_t)
        motor_speeds = []
        for T in [T0, T1, T2, T3]:
            if T > 0:
                omega = np.sqrt(T / self.thrust_coefficient)
                motor_speeds.append(clamp(omega, 400.0, 800.0))
            else:
                motor_speeds.append(400.0)
        
        return motor_speeds

    def control_loop(self):
        """Main control loop"""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        
        if dt < 0.001:  # Skip if dt too small
            return
        
        self.dt = dt
        
        # Apply feedback linearization control
        motor_speeds = self.feedback_linearization_control()
        
        # Publish motor commands
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = motor_speeds
        self.publisher_.publish(cmd)
        
        self.last_time = now
        
        # Logging
        self.get_logger().info(f"Pos: [{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}] | "
                              f"Des: [{self.desired_position[0]:.2f}, {self.desired_position[1]:.2f}, {self.desired_position[2]:.2f}] | "
                              f"Att: [{self.euler_angles[0]:.1f}, {self.euler_angles[1]:.1f}, {self.euler_angles[2]:.1f}] | "
                              f"Motors: [{motor_speeds[0]:.0f}, {motor_speeds[1]:.0f}, {motor_speeds[2]:.0f}, {motor_speeds[3]:.0f}]")
        
        # Publish reference position for plotting
        ref_msg = PointStamped()
        ref_msg.header.stamp = now.to_msg()
        ref_msg.point.x = self.desired_position[0]
        ref_msg.point.y = self.desired_position[1]
        ref_msg.point.z = self.desired_position[2]
        self.ref_pub.publish(ref_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FeedbackLinearizationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
