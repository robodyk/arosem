"""Integrate 2D odometry measurements in time, correctly handling covariance updates."""

import copy
import dataclasses
import unittest
from collections import deque
from threading import Lock
from typing import Optional, List, Deque, Dict, Tuple

import numpy as np

import rospy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from ros_numpy import msgify, numpify
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from aro_localization.utils import forward, normalize_angle_symmetric, w2r, col_vector, row_vector, interpolate_poses


def cov6_to_cov3(cov: List[float]) -> np.ndarray:
    cov = np.array(cov).reshape((6, 6))
    # The covariance is basically inverse of the Hessian we want to use as costs. However, to retain the Hessian values
    # when shrinking the matrix from 6x6 to 3x3, we need to go to the Hessian space (by taking inverse), then cut out
    # the 3x3 submatrix, and then go back to the covariance space by taking another inverse. This is a bit hacky, but
    # it's not wrong. You just have to select values from which space you want to retain.
    hessian = np.linalg.inv(cov + np.eye(6) * 1e-15)
    cov3 = np.linalg.inv(hessian[np.ix_((0, 1, 5), (0, 1, 5))])
    # This is important - if the input was all zeros, we need to output all zeros again, but we added the 1e-15 before.
    cov3[cov3 <= 1e-14] = 0
    return cov3


def cov3_to_cov6(cov3: np.ndarray) -> List[float]:
    cov = np.zeros((6, 6))
    cov[np.ix_((0, 1, 5), (0, 1, 5))] = cov3
    return cov.ravel().tolist()


@dataclasses.dataclass(init=False)
class OdomItem:
    """One received odometry item (measurement)."""
    stamp: rospy.Time
    """Timestamp of the measurement"""

    pos: np.ndarray
    """Absolute position in the odometry frame (Numpy array 3)."""

    twist: np.ndarray
    """Velocity in body frame (Numpy array 3)."""

    cov: np.ndarray
    """Covariance of the velocity (Numpy array 3x3)."""

    msg: Odometry
    """The Odometry message."""

    def __init__(self, msg: Odometry):
        self.stamp = msg.header.stamp
        _, _, yaw = euler_from_quaternion(numpify(msg.pose.pose.orientation))
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        self.twist = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
        self.cov = cov6_to_cov3(msg.twist.covariance)
        self.msg = copy.deepcopy(msg)


@dataclasses.dataclass
class IntegratedOdom:
    """Result of odometry integration at a single time step."""

    twist: np.ndarray
    """Relative motion from previous sample (Numpy array 3)."""

    twist_cov: np.ndarray
    """Relative motion covariance covariance (Numpy array 3x3)"""

    absolute_pos: Optional[np.ndarray] = None
    """Position recorded from the input odometry. (Numpy array 3)."""

    integrated_pos: Optional[np.ndarray] = None
    """Position integrated by the integrator by starting in origin and applying twist (Numpy array 3)."""


class OdomIntegrator:
    """Integrate 2D odometry measurements in time, correctly handling covariance updates.

    The standard mode of use is calling :meth:`add` whenever an odometry message is received, and calling :meth:`sample`
    when the interpolated measurements are required. It will return a short history of samples, each on in a time for
    which :meth:`sample` was called.
    """

    def __init__(self, history_length=rospy.Duration(3)):
        """
        :param history_length: Length of history window in seconds.
        """

        self.history_length = history_length
        self.items: Deque[OdomItem] = deque()
        """Measurements."""

        self.sample_times: Deque[rospy.Time] = deque()
        """Times in which the odometry has been sampled."""

        self.integrated_pos: Deque[np.ndarray] = deque()
        """Absolute position integrated from the relative odometry measurements."""

        self._mutex = Lock()

    def _prune_old_items(self, now: rospy.Time) -> None:
        """Keep the history of measurements and sample times within the bounds of the history window."""
        while len(self.items) > 0 and self.items[0].stamp + self.history_length < now:
            self.items.popleft()
        while len(self.sample_times) > 0 and self.sample_times[0] + self.history_length < now:
            self.sample_times.popleft()
            self.integrated_pos.popleft()

    def add(self, msg: Odometry, now: rospy.Time) -> None:
        """Add a new odometry measurement.

        :param msg: The measurement to be added. Only the twist and its covariance are integrated, as well as timestamp.
                    The interpretation of the twist is that it is the velocity derived from relative pose change
                    between the previous measurement time and the time of this measurement.
        :param now: Current time (used for pruning old odometry items).
        """
        with self._mutex:
            self._prune_old_items(now)
            self.items.append(OdomItem(msg))

    def apply_z_as_ekf_control(self, pos: np.ndarray, cov:np.ndarray, z_pos: np.ndarray, z_cov: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """Update integrated position (pos, cov) with another measurement of relative odometry (z_pos, z_cov).
        The function is basically application of one step of a Kalman filter with no measurements and z_pos as control.

        :param pos: So far integrated position (Numpy array 3).
        :param cov: Covariance of the so far integrated position (starting at 0 at the beginning) (Numpy array 3x3).
        :param z_pos: Relative odometry measurement (Numpy array 3).
        :param z_cov: Covariance of z_pos (Numpy array 3x3).
        :return: Updated position pos (Numpy array 3) and covariance cov (Numpy array 3x3).
        """
        d_x, d_y, d_yaw = z_pos
        s_yaw, c_yaw = np.sin(pos[2]), np.cos(pos[2])

        pos = pos + np.array([
            d_x * c_yaw - d_y * s_yaw,
            d_x * s_yaw + d_y * c_yaw,
            d_yaw
        ])
        pos[2] = normalize_angle_symmetric(pos[2])

        # Jacobian of g with respect to the state
        jac = np.array([
            [1.0, 0.0, -d_x * s_yaw + d_y * c_yaw],
            [0.0, 1.0, d_x * c_yaw - d_y * s_yaw],
            [0.0, 0.0, 1.0]
        ])
        cov = jac @ cov @ jac.T + z_cov

        return pos, cov

    def sample(self, stamp: rospy.Time) -> Dict[rospy.Time, IntegratedOdom]:
        """Get the relative 2D motion since last reset and its covariance and reset the integrator to integrate from
        scratch again.

        :param stamp: If set, instructs the integrator to integrate a bit beyond the last odometry measurement - i.e.
                      apply the last known velocity vector for time from the last odometry measurement to `stamp`.
        :returns: Timestamp-keyed dictionary of IntegratedOdom computed from the saved measurement history.
                  Returns empty dict if no odometry measurement is available.
        """
        with self._mutex:
            self._prune_old_items(stamp)
            self.sample_times.append(stamp)
            self.sample_times = deque(sorted(self.sample_times))
            self.integrated_pos.append(np.zeros((3,)))

            if len(self.items) <= 1 or len(self.sample_times) <= 1:
                return {}

            samples = {}
            for t in range(len(self.sample_times) - 1):
                t1 = self.sample_times[t]
                t2 = self.sample_times[t + 1]
                # TODO make more efficient?
                prev_is = [i for i in range(len(self.items)) if self.items[i].stamp <= t1]
                next_is = [i for i in range(len(self.items)) if self.items[i].stamp >= t2]
                if len(prev_is) == 0 or len(next_is) == 0:
                    continue
                min_i = prev_is[-1]
                max_i = next_is[0]
                assert min_i <= max_i
                twist = np.zeros((3,))
                twist_cov = np.zeros((3, 3))
                abs_pos = np.zeros((3,))
                i = min_i
                if self.items[i].stamp < t1:
                    r = 1 - (self.items[i + 1].stamp - t1).to_sec() / (self.items[i + 1].stamp - self.items[i].stamp).to_sec()
                    p_start = interpolate_poses(1 - r, self.items[i].pos, r, self.items[i + 1].pos)
                    p_end = self.items[i + 1].pos
                    dt = (self.items[i + 1].stamp - self.items[i].stamp).to_sec()
                    twist = w2r(p_end, p_start)
                    twist_cov = ((1 - r) * self.items[i].cov + r * self.items[i + 1].cov) * dt
                    abs_pos = p_start
                    i += 1
                i += 1
                has_exact_end = False
                while i < len(self.items) and self.items[i].stamp <= t2:
                    p_start = self.items[i - 1].pos
                    p_end = self.items[i].pos
                    dt = (self.items[i].stamp - self.items[i - 1].stamp).to_sec()
                    p = w2r(p_end, p_start)
                    c = self.items[i].cov * dt
                    twist, twist_cov = self.apply_z_as_ekf_control(twist, twist_cov, p, c)
                    abs_pos = p_end
                    if self.items[i].stamp == t2:
                        has_exact_end = True
                    i += 1
                if not has_exact_end:
                    i = max_i - 1
                    r = 1 - (self.items[i + 1].stamp - t2).to_sec() / (self.items[i + 1].stamp - self.items[i].stamp).to_sec()
                    p_start = self.items[i].pos
                    p_end = interpolate_poses(1 - r, self.items[i].pos, r, self.items[i + 1].pos)
                    dt = (self.items[i + 1].stamp - self.items[i].stamp).to_sec()
                    p = w2r(p_end, p_start)
                    c = ((1 - r) * self.items[i].cov + r * self.items[i + 1].cov) * dt
                    twist, twist_cov = self.apply_z_as_ekf_control(twist, twist_cov, p, c)
                    abs_pos = p_end
                self.integrated_pos[t + 1] = forward(twist[:, np.newaxis], self.integrated_pos[t])[:, 1]
                samples[t2] = IntegratedOdom(twist, twist_cov, abs_pos, self.integrated_pos[t + 1])

            return samples


class TestOdomIntegrator(unittest.TestCase):
    def test_odom_integrator_straight(self):
        dt = 0.2

        # ICP measurements (v_x,v_y,v_yaw)
        u_k = np.array([[1.0, 0.0, 0.0],  # k=1
                        [2.0, 0.0, 0],  # k=2
                        [3.0, 0.0, 0],  # k=3
                        [4.0, 0.0, 0],  # k=4
                        [5.0, 0.0, 0]])  # k=5

        p_k = forward(u_k.T * dt, (0.0, 0.0, 0.0)).T

        # ICP noise covariance matrix (should be different for different times)
        R_k = np.array([
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        ])

        odom = Odometry()
        integrator = OdomIntegrator()

        self.assertEqual(len(integrator.items), 0)
        self.assertEqual(len(integrator.sample_times), 0)

        odom.twist.covariance = cov3_to_cov6(0.1 * np.eye(3))
        integrator.add(odom, rospy.Time())

        self.assertEqual(len(integrator.items), 1)
        self.assertEqual(len(integrator.sample_times), 0)

        for i in range(u_k.shape[0]):
            odom.header.stamp += rospy.Duration(dt)
            odom.pose.pose.position.x = p_k[i + 1, 0]
            odom.pose.pose.position.y = p_k[i + 1, 1]
            odom.pose.pose.orientation = msgify(Quaternion, quaternion_from_euler(0, 0, p_k[i + 1, 2]))
            odom.twist.twist.linear.x = u_k[i, 0]
            odom.twist.twist.linear.y = u_k[i, 1]
            odom.twist.twist.angular.z = u_k[i, 2]
            odom.twist.covariance = cov3_to_cov6(R_k[i, :, :] / dt)
            integrator.add(odom, odom.header.stamp)
            self.assertEqual(len(integrator.items), i + 2)

        self.assertEqual(len(integrator.items), u_k.shape[0] + 1)

        samples = integrator.sample(rospy.Time(0))
        self.assertEqual(len(integrator.sample_times), 1)
        self.assertEqual(len(samples), 0)

        samples = integrator.sample(rospy.Time(0.45))
        self.assertEqual(len(integrator.sample_times), 2)
        self.assertEqual(len(samples), 1)
        self.assertIn(rospy.Time(0.45), samples)
        s1 = samples[rospy.Time(0.45)]
        p, cov, ap, ip = s1.twist, s1.twist_cov, s1.absolute_pos, s1.integrated_pos
        self.assertTrue(np.allclose(p, [0.75, 0., 0.0]))
        self.assertTrue(np.allclose(cov, [[0.3, 0., 0.], [0., 0.3325, 0.07], [0., 0.07, 0.3]]))
        self.assertTrue(np.allclose(p, ap))
        self.assertTrue(np.allclose(p, ip))

        samples = integrator.sample(rospy.Time(1.0))
        self.assertEqual(len(integrator.sample_times), 3)
        self.assertEqual(len(samples), 2)

        self.assertIn(rospy.Time(0.45), samples)
        s1 = samples[rospy.Time(0.45)]
        p, cov, ap, ip = s1.twist, s1.twist_cov, s1.absolute_pos, s1.integrated_pos
        self.assertTrue(np.allclose(p, [0.75, 0., 0.0]))
        self.assertTrue(np.allclose(cov, [[0.3, 0., 0.], [0., 0.3325, 0.07], [0., 0.07, 0.3]]))
        self.assertTrue(np.allclose(p, ap))
        self.assertTrue(np.allclose(p, ip))

        self.assertIn(rospy.Time(1), samples)
        s2 = samples[rospy.Time(1)]
        p, cov, ap, ip = s2.twist, s2.twist_cov, s2.absolute_pos, s2.integrated_pos
        self.assertTrue(np.allclose(p, [2.25, 0, 0]))
        self.assertTrue(np.allclose(cov, [[0.3, 0., 0.], [0., 0.724, 0.28], [0., 0.28, 0.3]]))
        self.assertTrue(np.allclose(ap, p_k[-1, :]))
        self.assertTrue(np.allclose(ip, p_k[-1, :]))

        pos = forward(np.array([s1.twist, s2.twist]).T, (0.0, 0.0, 0.0)).T
        self.assertTrue(np.allclose(p_k[-1, :], pos[-1, :]))

        pos, cov = integrator.apply_z_as_ekf_control(np.zeros((3,)), np.zeros((3, 3)), s1.twist, s1.twist_cov)
        pos, cov = integrator.apply_z_as_ekf_control(pos, cov, s2.twist, s2.twist_cov)
        self.assertTrue(np.allclose(p_k[-1, :], pos))

    def test_odom_integrator_turn(self):
        dt = 0.2

        # ICP measurements (v_x,v_y,v_yaw)
        u_k = np.array([[1.0, 0.0, 0.0],  # k=1
                        [2.0, 0.0, np.pi / 2],  # k=2
                        [3.0, 0.0, np.pi / 2],  # k=3
                        [4.0, 0.0, np.pi],  # k=4
                        [5.0, 0.0, np.pi]])  # k=5

        p_k = forward(u_k.T * dt, (0.0, 0.0, 0.0)).T

        # ICP noise covariance matrix (should be different for different times)
        R_k = np.array([
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 5, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        ])

        odom = Odometry()
        integrator = OdomIntegrator()

        self.assertEqual(len(integrator.items), 0)
        self.assertEqual(len(integrator.sample_times), 0)

        odom.twist.covariance = cov3_to_cov6(0.1 * np.eye(3))
        integrator.add(odom, rospy.Time())

        self.assertEqual(len(integrator.items), 1)
        self.assertEqual(len(integrator.sample_times), 0)

        for i in range(u_k.shape[0]):
            odom.header.stamp += rospy.Duration(dt)
            odom.pose.pose.position.x = p_k[i + 1, 0]
            odom.pose.pose.position.y = p_k[i + 1, 1]
            odom.pose.pose.orientation = msgify(Quaternion, quaternion_from_euler(0, 0, p_k[i + 1, 2]))
            odom.twist.twist.linear.x = u_k[i, 0]
            odom.twist.twist.linear.y = u_k[i, 1]
            odom.twist.twist.angular.z = u_k[i, 2]
            odom.twist.covariance = cov3_to_cov6(R_k[i, :, :])
            integrator.add(odom, odom.header.stamp)
            self.assertEqual(len(integrator.items), i + 2)

        self.assertEqual(len(integrator.items), u_k.shape[0] + 1)

        samples = integrator.sample(rospy.Time(0))
        self.assertEqual(len(integrator.sample_times), 1)
        self.assertEqual(len(samples), 0)

        samples = integrator.sample(rospy.Time(0.45))
        self.assertEqual(len(integrator.sample_times), 2)
        self.assertEqual(len(samples), 1)
        self.assertIn(rospy.Time(0.45), samples)
        s1 = samples[rospy.Time(0.45)]
        p, cov, ap, ip = s1.twist, s1.twist_cov, s1.absolute_pos, s1.integrated_pos
        self.assertTrue(np.allclose(p, [0.74265848, 0.04635255, 0.39269908]))
        self.assertTrue(np.allclose(cov, [[0.06008594, -0.00063532, -0.0018541],
                                          [-0.00063532, 0.31129659, 0.01370634],
                                          [-0.0018541, 0.01370634, 0.06]]))
        self.assertTrue(np.allclose(p, ap))
        self.assertTrue(np.allclose(p, ip))

        samples = integrator.sample(rospy.Time(1.0))
        self.assertEqual(len(integrator.sample_times), 3)
        self.assertEqual(len(samples), 2)

        self.assertIn(rospy.Time(0.45), samples)
        s1 = samples[rospy.Time(0.45)]
        p, cov, ap, ip = s1.twist, s1.twist_cov, s1.absolute_pos, s1.integrated_pos
        self.assertTrue(np.allclose(p, [0.74265848, 0.04635255, 0.39269908]))
        self.assertTrue(np.allclose(cov, [[0.06008594, -0.00063532, -0.0018541],
                                          [-0.00063532, 0.31129659, 0.01370634],
                                          [-0.0018541, 0.01370634, 0.06]]))
        self.assertTrue(np.allclose(p, ap))
        self.assertTrue(np.allclose(p, ip))

        self.assertIn(rospy.Time(1), samples)
        s2 = samples[rospy.Time(1)]
        p, cov, ap, ip = s2.twist, s2.twist_cov, s2.absolute_pos, s2.integrated_pos
        self.assertTrue(np.allclose(p, [1.87595678, 0.91185566, 1.49225651]))
        self.assertTrue(np.allclose(cov, [[0.08950667, -0.03691541, -0.03415136],
                                          [-0.03691541, 0.35418187, 0.04153584],
                                          [-0.03415136, 0.04153584, 0.06]]))
        self.assertTrue(np.allclose(ap, p_k[-1, :]))

        pos = forward(np.array([s1.twist, s2.twist]).T, (0.0, 0.0, 0.0)).T
        print(p_k[-1, :] - pos.T[:, -1], p_k[-1, :], pos[-1, :])
        self.assertTrue(np.allclose(p_k[-1, :], pos[-1, :], atol=0.05))

        pos, cov = integrator.apply_z_as_ekf_control(np.zeros((3,)), np.zeros((3, 3)), s1.twist, s1.twist_cov)
        pos, cov = integrator.apply_z_as_ekf_control(pos, cov, s2.twist, s2.twist_cov)
        self.assertTrue(np.allclose(p_k[-1, :], pos, atol=0.05))

    def test_odom_integrator_sample_all(self):
        dt = 0.2

        # ICP measurements (v_x,v_y,v_yaw)
        u_k = np.array([[1.0, 0.0, 0.0],  # k=1
                        [2.0, 0.0, 0],  # k=2
                        [3.0, 0.0, 0],  # k=3
                        [4.0, 0.0, 0],  # k=4
                        [5.0, 0.0, 0]])  # k=5

        p_k = forward(u_k.T * dt, (0.0, 0.0, 0.0)).T

        # ICP noise covariance matrix (should be different for different times)
        R_k = np.array([
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
            [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        ])

        odom = Odometry()
        integrator = OdomIntegrator()

        self.assertEqual(len(integrator.items), 0)
        self.assertEqual(len(integrator.sample_times), 0)

        integrator.add(odom, rospy.Time())

        self.assertEqual(len(integrator.items), 1)
        self.assertEqual(len(integrator.sample_times), 0)

        samples = integrator.sample(rospy.Time(0))
        self.assertEqual(len(integrator.sample_times), 1)
        self.assertEqual(len(samples), 0)

        for i in range(u_k.shape[0]):
            odom.header.stamp += rospy.Duration(dt)
            odom.pose.pose.position.x = p_k[i + 1, 0]
            odom.pose.pose.position.y = p_k[i + 1, 1]
            odom.pose.pose.orientation = msgify(Quaternion, quaternion_from_euler(0, 0, p_k[i + 1, 2]))
            odom.twist.twist.linear.x = u_k[i, 0]
            odom.twist.twist.linear.y = u_k[i, 1]
            odom.twist.twist.angular.z = u_k[i, 2]
            odom.twist.covariance = cov3_to_cov6(R_k[i, :, :] / dt)
            integrator.add(odom, odom.header.stamp)
            self.assertEqual(len(integrator.items), i + 2)

            samples = integrator.sample(odom.header.stamp)
            self.assertEqual(i + 1, len(samples))

            for t in range(len(samples)):
                sample = samples[sorted(samples.keys())[t]]
                self.assertTrue(np.allclose(u_k[t, :] * dt, sample.twist))
                self.assertTrue(np.allclose(R_k[t, :, :], sample.twist_cov))
                self.assertTrue(np.allclose(p_k[t + 1, :], sample.absolute_pos))
                self.assertTrue(np.allclose(p_k[t + 1, :], sample.integrated_pos))


if __name__ == '__main__':
    unittest.main()
