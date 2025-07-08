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

        # Feedback linearization gains
        # Position control gains (outer loop)
        self.kp_pos = np.array([5.0, 5.0, 10.0])  # [x, y, z]
        self.kd_pos = np.array([3.0, 3.0, 6.0])   # [x, y, z]
        
        # Attitude control gains (inner loop)
        self.kp_att = np.array([8.0, 8.0, 4.0])   # [roll, pitch, yaw]
        self.kd_att = np.array([2.0, 2.0, 1.0])   # [roll, pitch, yaw]

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
        
        # Calculate velocities (numerical differentiation)
        if self.previous_x is not None:
            self.velocity_x = (self.current_x - self.previous_x) / self.dt
            self.velocity_y = (self.current_y - self.previous_y) / self.dt
            self.velocity_z = (self.current_z - self.previous_z) / self.dt
            
            self.angular_velocity_x = (self.roll - self.previous_roll) / self.dt
            self.angular_velocity_y = (self.pitch - self.previous_pitch) / self.dt
            self.angular_velocity_z = (self.yaw - self.previous_yaw) / self.dt
        
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
        
        # Desired total thrust (along body z-axis)
        desired_thrust = self.mass * (virtual_control[2] + self.gravity)
        
        # Desired attitude angles from virtual control
        desired_roll = (virtual_control[0] * np.sin(self.yaw) - virtual_control[1] * np.cos(self.yaw)) / self.gravity
        desired_pitch = (virtual_control[0] * np.cos(self.yaw) + virtual_control[1] * np.sin(self.yaw)) / self.gravity
        desired_yaw = 0.0  # Keep yaw at 0 for simplicity
        
        # Clamp desired angles to reasonable limits
        desired_roll = clamp(desired_roll, -0.5, 0.5)  # ±30 degrees
        desired_pitch = clamp(desired_pitch, -0.5, 0.5)
        
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
        """
        # Control allocation matrix for X-configuration quadcopter
        # Motor arrangement:
        # 0: Front-right, 1: Back-left, 2: Front-left, 3: Back-right
        
        # Thrust force from each motor
        f0 = thrust/4 - tau_y/(2*self.arm_length) - tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        f1 = thrust/4 - tau_x/(2*self.arm_length) + tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        f2 = thrust/4 + tau_y/(2*self.arm_length) - tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        f3 = thrust/4 + tau_x/(2*self.arm_length) + tau_z/(4*self.drag_coefficient/self.thrust_coefficient)
        
        # Convert forces to motor speeds (omega = sqrt(F / k_thrust))
        motor_speeds = []
        forces = [f0, f1, f2, f3]
        
        for force in forces:
            if force > 0:
                omega = np.sqrt(force / self.thrust_coefficient)
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

[INFO] [1751980518.958211222] [Feedback_Linearization_Controller]: Pos: [-0.00, 0.00, 1.31] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980518.975521370] [Feedback_Linearization_Controller]: Pos: [-0.00, 0.00, 1.36] | Desired: [1.50, 1.50, 4.00] | Motors: [797, 800, 800, 797]
[INFO] [1751980518.987485331] [Feedback_Linearization_Controller]: Pos: [-0.00, 0.00, 1.42] | Desired: [1.50, 1.50, 4.00] | Motors: [684, 713, 713, 684]
[INFO] [1751980519.004033391] [Feedback_Linearization_Controller]: Pos: [0.00, -0.00, 1.48] | Desired: [1.50, 1.50, 4.00] | Motors: [715, 743, 743, 715]
[INFO] [1751980519.019506509] [Feedback_Linearization_Controller]: Pos: [0.00, -0.00, 1.54] | Desired: [1.50, 1.50, 4.00] | Motors: [652, 684, 683, 652]
[INFO] [1751980519.036416063] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.60] | Desired: [1.50, 1.50, 4.00] | Motors: [649, 681, 680, 648]
[INFO] [1751980519.053545971] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.67] | Desired: [1.50, 1.50, 4.00] | Motors: [636, 669, 668, 635]
[INFO] [1751980519.070615859] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.73] | Desired: [1.50, 1.50, 4.00] | Motors: [611, 646, 643, 608]
[INFO] [1751980519.090484177] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.79] | Desired: [1.50, 1.50, 4.00] | Motors: [638, 672, 670, 636]
[INFO] [1751980519.106932704] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.86] | Desired: [1.50, 1.50, 4.00] | Motors: [563, 602, 598, 559]
[INFO] [1751980519.125485425] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.92] | Desired: [1.50, 1.50, 4.00] | Motors: [592, 630, 626, 587]
[INFO] [1751980519.141491360] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 1.98] | Desired: [1.50, 1.50, 4.00] | Motors: [510, 555, 549, 503]
[INFO] [1751980519.159406688] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 2.04] | Desired: [1.50, 1.50, 4.00] | Motors: [541, 585, 578, 534]
[INFO] [1751980519.175713340] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 2.10] | Desired: [1.50, 1.50, 4.00] | Motors: [493, 542, 533, 483]
[INFO] [1751980519.192820071] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 2.17] | Desired: [1.50, 1.50, 4.00] | Motors: [492, 542, 533, 481]
[INFO] [1751980519.209773301] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 2.23] | Desired: [1.50, 1.50, 4.00] | Motors: [473, 526, 514, 459]
[INFO] [1751980519.227000045] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 2.28] | Desired: [1.50, 1.50, 4.00] | Motors: [478, 532, 519, 463]
[INFO] [1751980519.243184758] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.00, 2.34] | Desired: [1.50, 1.50, 4.00] | Motors: [440, 500, 485, 422]
[INFO] [1751980519.260181220] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.01, 2.40] | Desired: [1.50, 1.50, 4.00] | Motors: [453, 513, 496, 434]
[INFO] [1751980519.277029667] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.01, 2.45] | Desired: [1.50, 1.50, 4.00] | Motors: [445, 508, 489, 424]
[INFO] [1751980519.294116475] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.01, 2.51] | Desired: [1.50, 1.50, 4.00] | Motors: [451, 515, 495, 427]
[INFO] [1751980519.311458370] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.01, 2.56] | Desired: [1.50, 1.50, 4.00] | Motors: [460, 525, 503, 435]
[INFO] [1751980519.328559160] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.01, 2.61] | Desired: [1.50, 1.50, 4.00] | Motors: [444, 513, 488, 415]
[INFO] [1751980519.346155234] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.02, 2.66] | Desired: [1.50, 1.50, 4.00] | Motors: [458, 527, 501, 428]
[INFO] [1751980519.369382111] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.02, 2.71] | Desired: [1.50, 1.50, 4.00] | Motors: [512, 575, 551, 484]
[INFO] [1751980519.386466784] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.02, 2.75] | Desired: [1.50, 1.50, 4.00] | Motors: [512, 577, 551, 482]
[INFO] [1751980519.398116178] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.03, 2.79] | Desired: [1.50, 1.50, 4.00] | Motors: [438, 517, 483, 400]
[INFO] [1751980519.414172764] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.03, 2.84] | Desired: [1.50, 1.50, 4.00] | Motors: [426, 510, 472, 400]
[INFO] [1751980519.430961210] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.04, 2.88] | Desired: [1.50, 1.50, 4.00] | Motors: [444, 527, 488, 400]
[INFO] [1751980519.448000322] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.04, 2.92] | Desired: [1.50, 1.50, 4.00] | Motors: [451, 535, 494, 401]
[INFO] [1751980519.465227925] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.05, 2.96] | Desired: [1.50, 1.50, 4.00] | Motors: [459, 545, 502, 407]
[INFO] [1751980519.482362232] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.05, 3.00] | Desired: [1.50, 1.50, 4.00] | Motors: [463, 551, 505, 407]
[INFO] [1751980519.499973891] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.06, 3.04] | Desired: [1.50, 1.50, 4.00] | Motors: [486, 573, 527, 430]
[INFO] [1751980519.516787235] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.07, 3.07] | Desired: [1.50, 1.50, 4.00] | Motors: [473, 566, 515, 410]
[INFO] [1751980519.540588863] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.08, 3.11] | Desired: [1.50, 1.50, 4.00] | Motors: [567, 645, 602, 518]
[INFO] [1751980519.557299491] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.09, 3.13] | Desired: [1.50, 1.50, 4.00] | Motors: [559, 640, 594, 504]
[INFO] [1751980519.570894875] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.10, 3.16] | Desired: [1.50, 1.50, 4.00] | Motors: [470, 575, 512, 400]
[INFO] [1751980519.587714953] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.11, 3.19] | Desired: [1.50, 1.50, 4.00] | Motors: [512, 610, 550, 439]
[INFO] [1751980519.604238946] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.13, 3.22] | Desired: [1.50, 1.50, 4.00] | Motors: [533, 631, 570, 460]
[INFO] [1751980519.621484828] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.14, 3.25] | Desired: [1.50, 1.50, 4.00] | Motors: [541, 641, 577, 464]
[INFO] [1751980519.637476551] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.16, 3.27] | Desired: [1.50, 1.50, 4.00] | Motors: [531, 638, 568, 444]
[INFO] [1751980519.653523839] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.18, 3.30] | Desired: [1.50, 1.50, 4.00] | Motors: [547, 655, 583, 458]
[INFO] [1751980519.670560958] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.20, 3.32] | Desired: [1.50, 1.50, 4.00] | Motors: [571, 678, 605, 482]
[INFO] [1751980519.687606379] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.23, 3.34] | Desired: [1.50, 1.50, 4.00] | Motors: [590, 698, 623, 500]
[INFO] [1751980519.704445747] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.25, 3.35] | Desired: [1.50, 1.50, 4.00] | Motors: [608, 717, 640, 514]
[INFO] [1751980519.721277492] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.28, 3.37] | Desired: [1.50, 1.50, 4.00] | Motors: [628, 738, 659, 532]
[INFO] [1751980519.738625541] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.31, 3.38] | Desired: [1.50, 1.50, 4.00] | Motors: [652, 762, 682, 556]
[INFO] [1751980519.756079882] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.34, 3.39] | Desired: [1.50, 1.50, 4.00] | Motors: [677, 787, 705, 579]
[INFO] [1751980519.773671324] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.38, 3.40] | Desired: [1.50, 1.50, 4.00] | Motors: [703, 800, 730, 604]
[INFO] [1751980519.790763184] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.41, 3.40] | Desired: [1.50, 1.50, 4.00] | Motors: [732, 800, 758, 631]
[INFO] [1751980519.807657627] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.46, 3.40] | Desired: [1.50, 1.50, 4.00] | Motors: [764, 800, 789, 663]
[INFO] [1751980519.824459555] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.50, 3.40] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 699]
[INFO] [1751980519.841112852] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.55, 3.39] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 738]
[INFO] [1751980519.857819175] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.60, 3.37] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 780]
[INFO] [1751980519.875311261] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.65, 3.36] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.892663052] [Feedback_Linearization_Controller]: Pos: [-0.00, -0.71, 3.33] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.915016851] [Feedback_Linearization_Controller]: Pos: [-0.01, -0.77, 3.30] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.928306773] [Feedback_Linearization_Controller]: Pos: [-0.01, -0.83, 3.27] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.945307342] [Feedback_Linearization_Controller]: Pos: [-0.01, -0.89, 3.22] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.962379726] [Feedback_Linearization_Controller]: Pos: [-0.01, -0.96, 3.17] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.979097282] [Feedback_Linearization_Controller]: Pos: [-0.01, -1.02, 3.11] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980519.996262244] [Feedback_Linearization_Controller]: Pos: [-0.01, -1.09, 3.05] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 400, 800, 800]
[INFO] [1751980520.013352409] [Feedback_Linearization_Controller]: Pos: [-0.01, -1.15, 2.98] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980520.030976440] [Feedback_Linearization_Controller]: Pos: [-0.02, -1.22, 2.90] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980520.047370146] [Feedback_Linearization_Controller]: Pos: [-0.02, -1.28, 2.82] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]
[INFO] [1751980520.064479454] [Feedback_Linearization_Controller]: Pos: [-0.02, -1.34, 2.73] | Desired: [1.50, 1.50, 4.00] | Motors: [800, 800, 800, 800]



if __name__ == '__main__':
    main()
