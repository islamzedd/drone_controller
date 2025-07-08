# ex02_feedback_linearization_controller.py
"""
ROS 2 (Python rclpy) node implementing a **feedback‑linearisation controller** for a quadrotor.

Goals
-----
1.  Cancel the dominant nonlinear couplings of the full 6‑DOF rigid‑body model.
2.  Expose *virtual* linear inputs `(v_z, v_phi, v_theta, v_psi)` that
    can be handled with simple PD gains.
3.  Convert the required body forces/torques **u = [u0,u1,u2,u3]^T** to
    individual rotor squared‑speeds `w_i²` via the standard allocation
    matrix and publish them to the low‑level mixer / ESC interface.

The node subscribes to:
* `/odometry` (`nav_msgs/Odometry`) – current pose & twist.
* `/reference` (custom `quad_interfaces/msg/Reference`) – desired
  position (x,y,z) + yaw (ψ) and their first derivatives.

It publishes:
* `/motor_speeds` (`std_msgs/Float32MultiArray`) – four rotor speeds in
  **rad s⁻¹**.

Adjust the topic names or the message types to match your stack.  The
math follows the derivation we just walked through.
"""

import math
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

# === Utilities =============================================================

def euler_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Return (roll φ, pitch θ, yaw ψ) from quaternion (x, y, z, w)."""
    # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)  # use 90° if out of range
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# === Controller Node =======================================================

class FeedbackLinearisationController(Node):
    """ROS 2 node providing feedback‑linearised attitude/position control."""

    def __init__(self):
        super().__init__("feedback_linearisation_controller")

        # ─── Parameters (can also be declared dynamically) ─────────────────
        self.declare_parameter("mass", 1.50)          # kg
        self.declare_parameter("Ixx", 0.02)           # kg·m²
        self.declare_parameter("Iyy", 0.02)
        self.declare_parameter("Izz", 0.04)
        self.declare_parameter("arm_length", 0.20)    # m (distance from CG to rotor)
        self.declare_parameter("thrust_coeff", 8.54858e-6)  # N·s² (b)
        self.declare_parameter("drag_coeff", 1.6e-7)        # N·m·s² (d)
        self.declare_parameter("gravity", 9.80665)

        # ─── PD gains for the *virtual* linear system ──────────────────────
        # Translational (for v_z, ax, ay) – we actually close the outer loop
        # only on z (altitude) because x/y require mapping through attitude.
        self.declare_parameter("Kp_z", 4.0)
        self.declare_parameter("Kd_z", 3.0)
        # Rotational virtual inputs
        self.declare_parameter("Kp_phi", 4.0)
        self.declare_parameter("Kd_phi", 1.2)
        self.declare_parameter("Kp_theta", 4.0)
        self.declare_parameter("Kd_theta", 1.2)
        self.declare_parameter("Kp_psi", 2.5)
        self.declare_parameter("Kd_psi", 0.8)

        # ─── Topics ---------------------------------------------------------
        self.sub_odom = self.create_subscription(Odometry, "/odometry", self.cb_odom, 10)
        # Replace with your reference message type; here use Vector3 for (x,y,z) and w for yaw
        self.sub_ref = self.create_subscription(Vector3, "/reference", self.cb_reference, 10)

        self.pub_motors = self.create_publisher(Float32MultiArray, "/motor_speeds", 10)

        # ─── State variables ───────────────────────────────────────────────
        self.pose = np.zeros(6)   # x, y, z, φ, θ, ψ
        self.vel = np.zeros(6)    # ẋ, ẏ, ż, φ̇, θ̇, ψ̇

        self.ref_pos = np.zeros(4)   # x_d, y_d, z_d, ψ_d
        self.ref_vel = np.zeros(4)   # ẋ_d, ẏ_d, ż_d, ψ̇_d

        # Update timer (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)

    # ────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────────────────
    def cb_odom(self, msg: Odometry):
        # Position
        self.pose[0] = msg.pose.pose.position.x
        self.pose[1] = msg.pose.pose.position.y
        self.pose[2] = msg.pose.pose.position.z

        # Orientation
        q = msg.pose.pose.orientation
        self.pose[3:6] = euler_from_quaternion(q.x, q.y, q.z, q.w)

        # Linear velocities in world frame
        self.vel[0] = msg.twist.twist.linear.x
        self.vel[1] = msg.twist.twist.linear.y
        self.vel[2] = msg.twist.twist.linear.z

        # Angular velocities in body frame
        self.vel[3] = msg.twist.twist.angular.x
        self.vel[4] = msg.twist.twist.angular.y
        self.vel[5] = msg.twist.twist.angular.z

    def cb_reference(self, msg: Vector3):
        # NOTE: using Vector3 (x,y,z) with "z" field as yaw (ψ) for brevity
        # Replace by proper message for production use.
        self.ref_pos[0] = msg.x
        self.ref_pos[1] = msg.y
        self.ref_pos[2] = msg.z  # altitude setpoint
        # ref_pos[3] = ψ_d – yaw; stuff it via msg.y (?) or extend message.
        # For demonstration, keep yaw at zero.

    # ────────────────────────────────────────────────────────────────────────
    # Main control loop
    # ────────────────────────────────────────────────────────────────────────
    def control_loop(self):
        # --- Parameters ----------------------------------------------------
        m = self.get_parameter("mass").get_parameter_value().double_value
        Ixx = self.get_parameter("Ixx").get_parameter_value().double_value
        Iyy = self.get_parameter("Iyy").get_parameter_value().double_value
        Izz = self.get_parameter("Izz").get_parameter_value().double_value
        g = self.get_parameter("gravity").get_parameter_value().double_value
        l = self.get_parameter("arm_length").get_parameter_value().double_value
        b = self.get_parameter("thrust_coeff").get_parameter_value().double_value
        d = self.get_parameter("drag_coeff").get_parameter_value().double_value

        # Gains
        Kp_z = self.get_parameter("Kp_z").get_parameter_value().double_value
        Kd_z = self.get_parameter("Kd_z").get_parameter_value().double_value
        Kp_phi = self.get_parameter("Kp_phi").get_parameter_value().double_value
        Kd_phi = self.get_parameter("Kd_phi").get_parameter_value().double_value
        Kp_theta = self.get_parameter("Kp_theta").get_parameter_value().double_value
        Kd_theta = self.get_parameter("Kd_theta").get_parameter_value().double_value
        Kp_psi = self.get_parameter("Kp_psi").get_parameter_value().double_value
        Kd_psi = self.get_parameter("Kd_psi").get_parameter_value().double_value

        # -------------------------------------------------------------------
        # 1. Outer‑loop position control (here only altitude + crude xy)
        # -------------------------------------------------------------------
        # Altitude PD → virtual v_z (desired ẍ_z)
        pos_z_err = self.ref_pos[2] - self.pose[2]
        vel_z_err = self.ref_vel[2] - self.vel[2]
        v_z = Kp_z * pos_z_err + Kd_z * vel_z_err

        # Very simple x/y → desired roll/pitch via small‑angle approximation.
        pos_x_err = self.ref_pos[0] - self.pose[0]
        pos_y_err = self.ref_pos[1] - self.pose[1]
        # Map desired accelerations to φ, θ assuming small angles.
        ax_des = 1.5 * pos_x_err - 0.8 * self.vel[0]
        ay_des = 1.5 * pos_y_err - 0.8 * self.vel[1]
        # Desired roll, pitch (rad) using a = g * tan(angle)
        phi_des = ay_des / g
        theta_des = -ax_des / g

        # -------------------------------------------------------------------
        # 2. Inner‑loop attitude PD → virtual v_phi, v_theta, v_psi
        # -------------------------------------------------------------------
        v_phi = Kp_phi * (phi_des - self.pose[3]) + Kd_phi * (0.0 - self.vel[3])
        v_theta = Kp_theta * (theta_des - self.pose[4]) + Kd_theta * (0.0 - self.vel[4])
        v_psi = Kp_psi * (0.0 - self.pose[5]) + Kd_psi * (0.0 - self.vel[5])

        # -------------------------------------------------------------------
        # 3. Feedback‑linearisation control laws to compute u0‑u3
        # -------------------------------------------------------------------
        phi, theta, psi = self.pose[3:6]
        phi_dot, theta_dot, psi_dot = self.vel[3:6]

        # Decouple translational thrust (u0)
        u0 = m * (g + v_z) / (math.cos(phi) * math.cos(theta) + 1e-6)

        # Body torques (u1,u2,u3)
        u1 = -(Iyy - Izz) * theta_dot * psi_dot + Ixx * v_phi
        u2 = -(Izz - Ixx) * phi_dot * psi_dot + Iyy * v_theta
        u3 = -(Ixx - Iyy) * phi_dot * theta_dot + Izz * v_psi

        # -------------------------------------------------------------------
        # 4. Convert body force/torque to individual rotor speeds
        # -------------------------------------------------------------------
        # Allocation matrix B (maps rotor thrusts to body wrench).
        # For an X‑configuration quadrotor (front rotor: 1, left: 2, back: 3, right: 4)
        #
        #    [ b  b  b  b ]           = u0
        #    [ 0  b l  0 -b l ]       = u1
        # B = [ -b l  0  b l  0 ] ... = u2
        #    [  d -d  d -d ]          = u3
        # Multiply by w_i² to get forces (thrust in +z body frame).

        B = np.array([
            [ b,  b,  b,  b ],
            [ 0,  b*l, 0, -b*l ],
            [ -b*l, 0, b*l, 0 ],
            [  d, -d,  d, -d ]
        ])

        u_vec = np.array([u0, u1, u2, u3])

        # Solve for squared speeds (least squares in case of saturation / model mismatch)
        try:
            w_sq = np.linalg.solve(B, u_vec)
        except np.linalg.LinAlgError:
            # Fallback to pseudo‑inverse if singular
            w_sq = np.linalg.pinv(B) @ u_vec

        # Enforce physical limits
        w_sq = np.clip(w_sq, 0.0, 1e7)  # upper bound roughly (10 kRPM)^2
        w = np.sqrt(w_sq)

        # -------------------------------------------------------------------
        # 5. Publish motor speeds
        # -------------------------------------------------------------------
        msg = Float32MultiArray()
        msg.data = w.astype(float).tolist()
        self.pub_motors.publish(msg)

    # ────────────────────────────────────────────────────────────────────


def main():
    rclpy.init()
    node = FeedbackLinearisationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
