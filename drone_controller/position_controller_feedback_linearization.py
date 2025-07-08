import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PointStamped
from actuator_msgs.msg import Actuators
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler_scipy(w, x, y, z):
    r = R.from_quat([x, y, z, w])
    return r.as_euler('xyz', degrees=True)


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)


def generate_square_waypoints(size=2.0, height=2.0):
    half = size / 2.0
    return [[ half,  half, height],[-half,  half, height],[-half, -half, height],[ half, -half, height],[ half,  half, height]]


def generate_figure_eight_waypoints(num_points=7, A=2.0, B=2.0, H=2.0):
    t_vals = np.linspace(0, 2 * np.pi, num_points)
    return [[A*np.sin(t), B*np.sin(t)*np.cos(t), H] for t in t_vals]


def generate_cool_acrobatic_waypoints(num_points=300, duration=15.0, offset=(5,5,3), speed=1.0,
    amp_x1=4.0, amp_x2=2.0, amp_y1=3.0, amp_y2=1.5, amp_z1=1.8, amp_z2=0.5):
    waypoints=[]
    t_vals=np.linspace(0, duration, num_points)
    x_off,y_off,z_off=offset
    for t in t_vals:
        tf=speed*t
        x=amp_x1*np.sin(2*np.pi*0.3*tf)-amp_x2*np.sin(2*np.pi*0.6*tf)
        y=amp_y1*np.cos(2*np.pi*0.3*tf)-amp_y2*np.cos(2*np.pi*0.6*tf)
        z=2+amp_z1*np.sin(2*np.pi*0.5*tf)+amp_z2*np.sin(2*np.pi*1.2*tf)
        waypoints.append([x+x_off, y+y_off, z+z_off])
    return waypoints

class PositionController(Node):
    def __init__(self):
        super().__init__('Position_Controller_FB')
        # States
        self.x = self.y = self.z = None
        self.phi = self.theta = self.psi = None
        self.last_time = self.get_clock().now()
        self.wp_idx = 0
        self.wp_threshold = 0.05
        # Trajectory
        self.trajectory = generate_cool_acrobatic_waypoints(num_points=200, duration=20)

        # Subs & pubs
        self.sub = self.create_subscription(PoseArray, '/world/quadcopter/pose/info', self.pose_cb, 10)
        self.pub_ctrl = self.create_publisher(Actuators, '/X3/gazebo/command/motor_speed', 10)
        self.pub_ref = self.create_publisher(PointStamped, '/drone/ref_pos', 10)

    def pose_cb(self, msg: PoseArray):
        if len(msg.poses)<2:
            return
        p = msg.poses[1]
        self.x, self.y, self.z = p.position.x, p.position.y, p.position.z
        self.phi, self.theta, self.psi = quaternion_to_euler_scipy(
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
        self.control_loop()

    def control_loop(self):
        if self.x is None: return
        # Next waypoint
        wp = self.trajectory[self.wp_idx]
        dist = np.linalg.norm([wp[0]-self.x, wp[1]-self.y, wp[2]-self.z])
        if dist<self.wp_threshold:
            self.wp_idx = (self.wp_idx+1) % len(self.trajectory)
            wp = self.trajectory[self.wp_idx]

        # timing
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt<1e-3: return

        # Feedback linearization for position
        # Desired accelerations: u = x_ddot + Kd*x_dot_err + Kp*x_err
        # Here we approximate x_dot & y_dot by finite diff
        # Gains
        Kp = np.diag([5,5,8])
        Kd = np.diag([3,3,5])
        # Position error
        pos = np.array([self.x, self.y, self.z])
        vel = np.zeros(3)  # assume low-level handles
        pos_des = np.array(wp)
        acc_des = np.zeros(3)  # no feedforward accel
        acc_cmd = acc_des + Kd.dot(vel) + Kp.dot(pos_des-pos)

        # Convert acc_cmd to thrust and attitude setpoints
        # thrust = m*(g + acc_z)
        m = 1.0; g=9.81
        thrust = m*(g + acc_cmd[2])
        # desired pitch and roll: phi_cmd = (1/g)*(acc_x*sin(psi)-acc_y*cos(psi))
        phi_cmd = (1/g)*(acc_cmd[0]*np.sin(self.psi) - acc_cmd[1]*np.cos(self.psi))
        theta_cmd = (1/g)*(acc_cmd[0]*np.cos(self.psi) + acc_cmd[1]*np.sin(self.psi))

        # saturate
        thrust = clamp(thrust, 0, 20)
        phi_cmd = clamp(phi_cmd, -0.5, 0.5)
        theta_cmd = clamp(theta_cmd, -0.5, 0.5)

        # Attitude feedback linearization: simple PD
        Kp_att = np.array([4,4,1])
        Kd_att = np.array([2,2,0.5])
        err_att = np.array([phi_cmd-self.phi, theta_cmd-self.theta, 0-self.psi])
        # assume angular rates are measured or zero
        omega = np.zeros(3)
        u_att = Kp_att*err_att + Kd_att*(0-omega)

        # Motor mixing
        # [f1, f2, f3, f4] = mixing matrix * [thrust; u_phi; u_theta; u_psi]
        mix = np.array([[1,-1,-1,-1],[1,-1,1,1],[1,1,1,-1],[1,1,-1,1]])
        cmds = mix.dot([thrust, u_att[0], u_att[1], u_att[2]])
        cmds = np.clip(cmds, 0, 1000)

        # publish motors
        cmd = Actuators()
        cmd.header = Header(); cmd.header.stamp = now.to_msg()
        cmd.velocity = cmds.tolist()
        self.pub_ctrl.publish(cmd)

        # ref pos
        ref = PointStamped(); ref.header.stamp=self.get_clock().now().to_msg()
        ref.point.x,ref.point.y,ref.point.z = wp
        self.pub_ref.publish(ref)

        self.last_time = now
        self.get_logger().info(f"pos: {self.x:.2f},{self.y:.2f},{self.z:.2f} | wp: {wp} | thrust: {thrust:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = PositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
