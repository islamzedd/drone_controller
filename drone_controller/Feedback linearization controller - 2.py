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

def safe_arcsin(x):
    """Safely compute arcsin, clamping input to valid range"""
    return np.arcsin(clamp(x, -1.0, 1.0))

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

        # Physical parameters (adjust based on your quadcopter model)
        self.m = 1.0  # Mass (kg)
        self.g = 9.81  # Gravity (m/s²)
        self.l = 0.25  # Arm length (m)
        
        # Moments of inertia (kg·m²)
        self.Ixx = 0.01  
        self.Iyy = 0.01
        self.Izz = 0.02
        
        # State variables
        self.current_x = None
        self.current_y = None
        self.current_z = None
        
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        
        self.roll = None
        self.pitch = None
        self.yaw = None
        
        self.omega_x = 0.0  # Angular velocities
        self.omega_y = 0.0
        self.omega_z = 0.0
        
        # Previous values for numerical differentiation
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        self.prev_roll = None
        self.prev_pitch = None
        self.prev_yaw = None
        
        # Trajectory tracking
        self.current_wp_idx = 0
        self.wp_threshold = 0.05
        self.trajectory = generate_square_waypoints(size=3.0, height=4.0)
        
        # Control gains for feedback linearization
        # Position control gains
        self.kp_pos = np.array([10.0, 10.0, 15.0])  # [x, y, z]
        self.kd_pos = np.array([5.0, 5.0, 8.0])     # [x, y, z]
        
        # Attitude control gains
        self.kp_att = np.array([5.0, 5.0, 2.0])     # [roll, pitch, yaw]
        self.kd_att = np.array([1.0, 1.0, 0.5])     # [roll, pitch, yaw]
        
        # Time tracking
        self.last_time = self.get_clock().now()
        
        # Subscribers and publishers
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
        
        # Store previous values for velocity estimation
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        self.prev_z = self.current_z
        self.prev_roll = self.roll
        self.prev_pitch = self.pitch
        self.prev_yaw = self.yaw
        
        # Get current position and orientation
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
        
        self.feedback_linearization_control()

    def feedback_linearization_control(self):
        if self.current_z is None:
            return
        
        # Time step calculation
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.001:
            return
        
        # Check for valid current pose values
        if (np.isnan(self.current_x) or np.isnan(self.current_y) or np.isnan(self.current_z) or
            np.isnan(self.roll) or np.isnan(self.pitch) or np.isnan(self.yaw)):
            self.get_logger().warn("NaN in current pose, skipping control iteration")
            return
        
        # Estimate velocities using numerical differentiation
        if self.prev_x is not None and not np.isnan(self.prev_x):
            self.vel_x = (self.current_x - self.prev_x) / dt
            self.vel_y = (self.current_y - self.prev_y) / dt
            self.vel_z = (self.current_z - self.prev_z) / dt
            
            # Handle angle wraparound for angular velocities
            roll_diff = self.roll - self.prev_roll
            pitch_diff = self.pitch - self.prev_pitch
            yaw_diff = self.yaw - self.prev_yaw
            
            # Wrap angle differences to [-π, π]
            roll_diff = np.arctan2(np.sin(roll_diff), np.cos(roll_diff))
            pitch_diff = np.arctan2(np.sin(pitch_diff), np.cos(pitch_diff))
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            
            self.omega_x = roll_diff / dt
            self.omega_y = pitch_diff / dt
            self.omega_z = yaw_diff / dt
            
            # Check for NaN in angular velocities
            if np.isnan(self.omega_x) or np.isnan(self.omega_y) or np.isnan(self.omega_z):
                self.get_logger().warn("NaN in angular velocities, setting to zero")
                self.omega_x = self.omega_y = self.omega_z = 0.0
        else:
            # Initialize velocities to zero on first iteration
            self.vel_x = self.vel_y = self.vel_z = 0.0
            self.omega_x = self.omega_y = self.omega_z = 0.0
        
        # Waypoint tracking
        if self.current_wp_idx >= len(self.trajectory):
            self.current_wp_idx = 0
        
        desired_pos = self.trajectory[self.current_wp_idx]
        
        # Check if close to waypoint
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
        
        # Desired position and velocity (for trajectory tracking)
        x_d = np.array([desired_pos[0], desired_pos[1], desired_pos[2]])
        v_d = np.array([0.0, 0.0, 0.0])  # Zero desired velocity for waypoint tracking
        
        # Current position and velocity
        x_curr = np.array([self.current_x, self.current_y, self.current_z])
        v_curr = np.array([self.vel_x, self.vel_y, self.vel_z])
        
        # Position error
        e_pos = x_d - x_curr
        e_vel = v_d - v_curr
        
        # Desired accelerations from position control
        a_d = self.kp_pos * e_pos + self.kd_pos * e_vel
        
        # Feedback linearization for position control
        # Calculate desired thrust u0 with safety checks
        cos_roll = np.cos(self.roll)
        cos_pitch = np.cos(self.pitch)
        cos_product = cos_roll * cos_pitch
        
        # Avoid division by zero
        if abs(cos_product) < 0.1:  # Prevent near-zero division
            cos_product = 0.1 * np.sign(cos_product) if cos_product != 0 else 0.1
        
        u0_desired = self.m * (a_d[2] + self.g) / cos_product
        u0_desired = clamp(u0_desired, 0.1, 20.0)  # Limit thrust
        
        # Calculate desired roll and pitch angles with safety checks
        sin_yaw = np.sin(self.yaw)
        cos_yaw = np.cos(self.yaw)
        
        # Calculate arguments for arcsin and clamp them
        phi_arg = self.m * (a_d[0] * sin_yaw - a_d[1] * cos_yaw) / u0_desired
        phi_d = safe_arcsin(phi_arg)
        
        cos_phi_d = np.cos(phi_d)
        if abs(cos_phi_d) < 0.1:  # Prevent near-zero division
            cos_phi_d = 0.1 * np.sign(cos_phi_d) if cos_phi_d != 0 else 0.1
        
        theta_arg = self.m * (a_d[0] * cos_yaw + a_d[1] * sin_yaw) / (u0_desired * cos_phi_d)
        theta_d = safe_arcsin(theta_arg)
        
        # Clamp desired angles to reasonable limits
        phi_d = clamp(phi_d, -0.3, 0.3)      # ±17 degrees
        theta_d = clamp(theta_d, -0.3, 0.3)   # ±17 degrees
        psi_d = 0.0  # Desired yaw (can be modified for trajectory tracking)
        
        # Attitude error
        e_att = np.array([phi_d - self.roll, theta_d - self.pitch, psi_d - self.yaw])
        e_omega = np.array([0.0 - self.omega_x, 0.0 - self.omega_y, 0.0 - self.omega_z])
        
        # Check for NaN in attitude errors
        if np.any(np.isnan(e_att)) or np.any(np.isnan(e_omega)):
            self.get_logger().warn(f"NaN in attitude errors: e_att={e_att}, e_omega={e_omega}")
            self.get_logger().warn(f"phi_d={phi_d}, theta_d={theta_d}, roll={self.roll}, pitch={self.pitch}")
            self.get_logger().warn(f"omega_x={self.omega_x}, omega_y={self.omega_y}, omega_z={self.omega_z}")
            return
        
        # Desired angular accelerations from attitude control
        alpha_d = self.kp_att * e_att + self.kd_att * e_omega
        
        # Check for NaN in desired angular accelerations
        if np.any(np.isnan(alpha_d)):
            self.get_logger().warn(f"NaN in alpha_d: {alpha_d}")
            self.get_logger().warn(f"e_att={e_att}, e_omega={e_omega}")
            self.get_logger().warn(f"kp_att={self.kp_att}, kd_att={self.kd_att}")
            return
        
        # Feedback linearization for attitude control
        # Calculate control moments u1, u2, u3 with safety checks
        u1 = self.Ixx * alpha_d[0] + (self.Iyy - self.Izz) * self.omega_y * self.omega_z
        u2 = self.Iyy * alpha_d[1] + (self.Izz - self.Ixx) * self.omega_z * self.omega_x
        u3 = self.Izz * alpha_d[2] + (self.Ixx - self.Iyy) * self.omega_x * self.omega_y
        
        # Check for NaN in control moments
        if np.isnan(u1) or np.isnan(u2) or np.isnan(u3):
            self.get_logger().warn(f"NaN in control moments: u1={u1}, u2={u2}, u3={u3}")
            self.get_logger().warn(f"alpha_d={alpha_d}")
            self.get_logger().warn(f"omega_x={self.omega_x}, omega_y={self.omega_y}, omega_z={self.omega_z}")
            return
        
        # Convert control inputs to motor speeds
        # Motor mixing (X configuration)
        motor_thrust = u0_desired / 4.0  # Base thrust per motor
        
        # Convert moments to motor speed differences
        roll_comp = u1 / (4.0 * self.l)
        pitch_comp = u2 / (4.0 * self.l)
        yaw_comp = u3 / (4.0 * 0.01)  # Torque constant (adjust based on motor)
        
        # Calculate individual motor thrusts
        motor0_thrust = motor_thrust - roll_comp - pitch_comp + yaw_comp
        motor1_thrust = motor_thrust + roll_comp + pitch_comp + yaw_comp
        motor2_thrust = motor_thrust + roll_comp - pitch_comp - yaw_comp
        motor3_thrust = motor_thrust - roll_comp + pitch_comp - yaw_comp
        
        # Convert thrust to motor speeds (approximate relationship)
        # Ensure positive thrust values before square root
        k_t = 0.001  # Thrust coefficient (adjust based on motor/propeller)
        
        # Clamp thrusts to positive values
        motor0_thrust = max(motor0_thrust, 0.001)
        motor1_thrust = max(motor1_thrust, 0.001)
        motor2_thrust = max(motor2_thrust, 0.001)
        motor3_thrust = max(motor3_thrust, 0.001)
        
        motor0_speed = np.sqrt(motor0_thrust / k_t)
        motor1_speed = np.sqrt(motor1_thrust / k_t)
        motor2_speed = np.sqrt(motor2_thrust / k_t)
        motor3_speed = np.sqrt(motor3_thrust / k_t)
        
        # Convert to the expected range and clamp
        motor0 = clamp(motor0_speed + 636.0, 400.0, 800.0)
        motor1 = clamp(motor1_speed + 636.0, 400.0, 800.0)
        motor2 = clamp(motor2_speed + 636.0, 400.0, 800.0)
        motor3 = clamp(motor3_speed + 636.0, 400.0, 800.0)
        
        # Final safety check for NaN values
        if np.isnan(motor0) or np.isnan(motor1) or np.isnan(motor2) or np.isnan(motor3):
            self.get_logger().warn("NaN detected in motor commands, using safe defaults")
            motor0 = motor1 = motor2 = motor3 = 636.0
        
        # Publish motor commands
        cmd = Actuators()
        cmd.header = Header()
        cmd.header.stamp = now.to_msg()
        cmd.velocity = [motor0, motor1, motor2, motor3]
        self.publisher_.publish(cmd)
        
        self.last_time = now
        
        # Logging
        self.get_logger().info(f"Pos: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | "
                              f"Des: ({desired_pos[0]:.2f}, {desired_pos[1]:.2f}, {desired_pos[2]:.2f}) | "
                              f"Att: φ={np.degrees(self.roll):.1f}°, θ={np.degrees(self.pitch):.1f}°, ψ={np.degrees(self.yaw):.1f}° | "
                              f"Des Att: φ={np.degrees(phi_d):.1f}°, θ={np.degrees(theta_d):.1f}° | "
                              f"Motors: [{motor0:.0f}, {motor1:.0f}, {motor2:.0f}, {motor3:.0f}]")
        
        # Publish reference position for plotting
        ref_msg = PointStamped()
        ref_msg.header.stamp = self.get_clock().now().to_msg()
        ref_msg.point.x = desired_pos[0]
        ref_msg.point.y = desired_pos[1]
        ref_msg.point.z = desired_pos[2]
        self.ref_pub.publish(ref_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FeedbackLinearizationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
