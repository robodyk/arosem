"""
Basic implementation of ICP-based SLAM. The inputs are pointclouds from individual lidar scans and the outputs are
transformations between the scans and previous position. Also, a map can be created. There are several options that
influence the robustness and quality of the ICP odometry: whether the alignment is done in 2D (SE2) or 3D (SE3). Next,
errors between nearest neighbor points can be computed either as Euclidean distances (point_to_point) or as distances
along surface normals (point_to_plane). Finally, incoming scans can be aligned either only to previous scan
(frame_to_frame), or to a continually built map (frame_to_map). Computing the transformations in SE3 can be beneficial
even for 2D lidars if e.g. the robot can tilt a bit when accelerating and the scans get distorted (this happens on the
real Turtlebots).

Run this file as
    python3 -m unittest icp.py
to run the unit tests that check basic validity of your implementation.

Run this file as
    python3 icp.py
to run a few demos utilizing your implementation.

To run ICP SLAM with the full simulation, run the following inside singularity:
    roslaunch aro_exploration aro_slam_sim.launch

Documentation comments in this file use terminology 'Numpy array X' to denote numpy array with shape X.
So 'numpy array 3xN' means 2-dimensional numpy array with shape (3, N). (3 rows, N columns).
'numpy array 3' means 1-dimensional array with shape (3,). Please note that this sometimes creates grammatically weird
sentences.
"""

import unittest
from dataclasses import dataclass
from enum import Enum
from timeit import default_timer as timer
from typing import Optional, Sequence, Callable, Tuple, Any, List, Union

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.spatial import cKDTree

from aro_slam.utils import as_unstructured

try:
    import rospy
    logdebug = rospy.logdebug
    loginfo = rospy.loginfo
    logwarn = rospy.logwarn
    logerr = rospy.logerr
except ImportError:
    import sys
    def logdebug(*args, **kwargs): pass
    loginfo = print
    def logwarn(*args, **kwargs): print(*args, **kwargs, file=sys.stderr)
    def logerr(*args, **kwargs): print(*args, **kwargs, file=sys.stderr)

from aro_slam.clouds import e2p, normal, p2e, position, transform, DType


class Loss(Enum):
    """The type of geometrical loss computed for ICP."""

    point_to_plane = 'point_to_plane'
    """Geometrical loss computing distance of the point to the plane given by the corresponding point's normal."""

    point_to_point = 'point_to_point'
    """Geometrical loss computing point-to-point Euclidean distance."""


class AbsorientDomain(Enum):
    """The dimensionality of the absolute orientation problem."""

    SE2 = 'SE2'
    """Only solve for x, y, yaw coordinates."""

    SE3 = 'SE3'
    """Solve for x, y, z and 3D rotation."""


class Alignment(Enum):
    """What should new scans be aligned to?"""

    frame_to_frame = 'frame_to_frame'
    """Align new scans only to the previous scan."""

    frame_to_map = 'frame_to_map'
    """Align new scans to a continually built map."""





def index(seq: Sequence[Any], idx: Sequence[int]) -> List[Any]:
    """Square-bracket-like indexing for non-numpy sequences.
    :param seq: The input sequence.
    :param idx: The indices of input sequence to select.
    :return: The subset of `seq` with `idx` elements.
    """
    return [seq[i] for i in idx]


def sample(seq: Sequence[Any], n: int) -> Sequence[Any]:
    """Get a representative set of `n` samples from `seq`.
    :param seq: The input sequence. 
    :param n: How many samples to get.
    :return: `n` samples from `seq`.
    """
    if len(seq) == 0:
        return []
    idx = np.unique(np.linspace(0, len(seq) - 1, n, dtype=int))
    s = index(seq, idx)
    return s


def absolute_orientation(p: np.ndarray, q: np.ndarray, domain=AbsorientDomain.SE2) -> np.ndarray:
    r"""Find transform R, t between p and q such that the sum of squared distances

    .. math:: \sum_{i=0}^{N-1} ||R * p[:, i] + t - q[:, i]||^2

    is minimal.

    :param p: Points to align, Numpy array 3xN.
    :param q: Reference points to align to. These are already expected to be in the correct order given by the
              correspondence mapping. Numpy array 3xN.
    :param domain: SE2 or SE3.

    :return: Optimized SE(D) transform T = [R t; 0... 1]. For SE2, z translation, roll and pitch are zero.
             Numpy array 4x4.
    """
    assert p.shape == q.shape, 'Inputs must be same size.'
    assert p.shape[0] == 3
    assert p.shape[1] > 0

    # TODO HW 3: Implement absolute orientation for both SE2 and SE3 domains (follow the FIXME markers).

    # STEP 1: Center the points in p and q.
    p_centered = np.zeros_like(p)  # FIXME corresponds to p' from lectures
    q_centered = np.zeros_like(q)  # FIXME corresponds to q' from lectures


    # STEP 2: Compute optimal rotation R (directly from theta for SE2, or using SVD for SE3).
    # The rotation R^* is the minimizer of \sum_i ||R * p'[:, i] - q'[:, i]||^2 .
    # Hint: use https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html for SE3.
    # Notice: np.linalg.svd() returns H = U * diag(s) * V, not U * diag(s) * V'.
    R = np.eye(3)  # FIXME


    # Sometimes SVD returns a reflection matrix instead of rotation
    if np.isclose(np.linalg.det(R), -1.0):
        # FIXME in SE3 case, this is wrong. You should instead negate the last column of V when det(R) is -1
        R[:, 2] = -R[:, 2]

    if not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Rotation R, R'*R = I, det(R) = 1, could not be found.")

    # STEP 3: Finally, apply the formula to compute t from known R.
    # t^* = \argmin_t ||R^* * p~ + t − q~||^2 = q~ − R^* * p~
    t = np.zeros((3, 1))  # FIXME


    # STEP 4: Construct the final transform
    T = np.eye(4)
    T[:-1, :-1] = R
    T[:-1, -1:] = t

    return T


@dataclass
class IcpResult:
    """ICP registration result."""

    T: Optional[np.ndarray] = None
    """The aligning transform (Numpy array 4x4) or None if registration failed."""
    num_iters: Optional[int] = None
    """Number of iterations run."""
    correspondences: Optional[Sequence[int]] = None
    """Nearest neighbors scan matching (correspondences c(i) )."""
    inliers: Optional[np.ndarray] = None
    """Inlier mask, use x[inliers] or y[correspondences[inliers]]. Numpy int array M."""
    p_inliers: Optional[np.ndarray] = None
    """Aligned position array of inliers. Numpy array Mx3."""
    q_inliers: Optional[np.ndarray] = None
    """Reference position array of inliers. In general, not original y for point-to-plane loss. Numpy array Mx3."""
    mean_inlier_dist: Optional[float] = None
    """Mean point distance for inlier correspondences."""
    cov: Optional[np.ndarray] = None
    """Covariance of the alignment."""


def icp(p_struct: np.ndarray, q_struct: np.ndarray, q_index: Optional[cKDTree] = None, T0: Optional[np.ndarray] = None,
        max_iters=50, inlier_ratio=1.0, inlier_dist_mult=1.0, max_inlier_dist=float('inf'),
        loss=Loss.point_to_point, absorient_domain=AbsorientDomain.SE2) -> IcpResult:
    """Iterative closest point (ICP) algorithm, minimizing sum of squared
    point-to-point or point-to-plane distances.

    Input point clouds are structured Numpy arrays, with
    - position fields 'x', 'y', 'z', and
    - normal fields 'normal_x', 'normal_y', 'normal_z'.

    :param p_struct: Points to align, structured array. Numpy structured array N.
    :param q_struct: Reference points to align to. Numpy structured array M.
    :param q_index: Nearest neighbor search index for q_struct. KDTree, which can be queried with position(p_struct).
    :param T0: Initial transform estimate from SE(D), defaults to identity. Numpy array 4x4.
    :param max_iters: Maximum number of iterations.
    :param inlier_ratio: Ratio of inlier correspondences with the lowest distances for which we optimize the criterion
                         in given iteration. The inliers set may change each iteration.
    :param inlier_dist_mult: Multiplier of the maximum inlier distance found using inlier ratio above. It enlarges or
                             reduces the inlier set for optimization.
    :param max_inlier_dist: Maximum distance for inlier correspondences [m].
    :param loss: The geometrical loss to use.
    :param absorient_domain: Absolute orientation domain, SE(2) or SE(3).
    :return: IcpResult with:
      - Optimized transform from SE(D) as 4x4 Numpy array
      - mean inlier distance from the last iteration
      - correspondences (list of indices into q_struct such that p_struct[i] corresponds to q_struct[correspondences[i]]
      - boolean inlier mask from the last iteration (p_struct[inl] are the inlier points)
    """
    assert isinstance(p_struct, np.ndarray) and p_struct.shape[0] > 0
    assert isinstance(q_struct, np.ndarray) and q_struct.shape[0] > 0
    assert q_index is None or isinstance(q_index, cKDTree)
    assert T0 is None or (isinstance(T0, np.ndarray) and T0.shape == (4, 4))
    assert max_iters > 0
    assert 0.0 <= inlier_ratio <= 1.0
    assert 0.0 < inlier_dist_mult

    t = timer()

    # Extract positions of points from the structured input clouds.
    p = position(p_struct)
    q = position(q_struct)
    assert isinstance(q, np.ndarray)

    # If q_index is not passed, create the KD-tree. For best performance, you should reuse the index.
    if q_index is None:
        q_index = cKDTree(q)

    # Boolean inlier mask from current iteration.
    inl = np.zeros((p_struct.size,), dtype=bool)
    # Mean inlier distance history (to assess improvement).
    inl_errs = []

    # Mean inliers ratios (relative to number of points in a source cloud) history.
    inl_ratios = []

    # Correspondences c(i).
    c: Optional[np.ndarray] = None

    # p and q inliers
    p_inl: Optional[np.ndarray] = None
    q_inl: Optional[np.ndarray] = None

    # Extract normals for later use in point-to-plane.
    if loss == Loss.point_to_plane or 'normal_x' in p_struct.dtype.fields:
        q_normal = normal(q_struct)



    # STEP 0: The ICP algorithm starts here.

    # STEP 1: Initialize transform T by the given initial guess (or identity if no guess is available).
    T = T0 if T0 is not None else np.eye(4)

    for iter in range(max_iters):
        # TODO HW 3: Transform source points p_struct using last known T to align with reference points.
        # Hint: use imported function transform()


        # STEP 2: Solve nearest neighbors.
        # TODO HW 3: Find correspondences (Nearest Neighbors Search).
        # Find distances between source and reference point clouds and corresponding indexes.
        # Hint: use https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html .
        # For point-to-plane loss, create the virtual q^ points and recompute distances from those.

        # Distances of points from p to their nearest neighbors in q_index. Don't forget that point-to-plane computes
        # the distances differently.
        dists = np.zeros(len(p_struct))  # FIXME
        # Indices of nearest neighbors in q_index, correspondences c(i).
        c = np.random.randint(len(q_struct), size=p_struct.shape)  # FIXME



        # STEP 3: Outlier rejection by median thresholding.
        # Construct the set of inliers (median filter, already implemented here).
        d_max = np.percentile(dists, 100.0 * inlier_ratio)
        # Use fixed sample size for assessing improvement.
        inl = dists <= d_max
        inl_errs.append(dists[inl].mean())
        # Use adaptive sample size for optimization.
        d_max = inlier_dist_mult * d_max
        d_max = min(d_max, max_inlier_dist)
        inl = dists <= d_max
        inl_ratios.append(inl.mean())
        n_inliers = inl.sum()
        if n_inliers == 0:
            logwarn('Not enough inliers: %i.', n_inliers)
            break

        # STEP 4: Stop if inlier error does not change much for some time.
        # TODO HW 3: Stop the ICP loop when the inliers error does not change much (call `break` when converged)
        # Array inl_errs contains a history of errors, so look for changes in there.


        # STEP 5: Solve absolute orientation for inliers.
        p_inl = p[inl]
        q_inl = q[c[inl]]  # FIXME Change q_inl to the virtual points q^ for point-to-plane loss


        # Use the inlier points with found correspondences to compute updated T (already implemented here).
        try:
            T = absolute_orientation(p_inl.T, q_inl.T, domain=absorient_domain)
        except ValueError as ex:
            logerr('Absolute orientation failed: %s', ex)
            return IcpResult(num_iters=iter, correspondences=c, inliers=inl, p_inliers=p_inl, q_inliers=q_inl)
    else:
        # This else is executed only when the for loop did not stop using break. This means the stopping condition has
        # not been met, thus ICP did not converge.
        logwarn('Max iter. %i: inliers: %.2f, mean dist.: %.3g, max dist: %.3g.',
                max_iters, inl.mean(), inl_errs[-1], d_max)

    # Compute covariance of the alignment (already implemented here).
    # The covariance computation is only valid for point-to-plane losses, but we provide it even for point-to-point
    # to provide an idea about how geometrically constrained the scene is.
    # The computation is based on:
    # Barczyk, M. & Bonnabel, S. & Goulette, F. (2014). Observability, Covariance and Uncertainty of ICP Scan Matching.
    cov: Optional[np.ndarray] = None
    if 'normal_x' in p_struct.dtype.fields and inl.sum() > 0:
        p_inl = p[inl]
        normals = q_normal[c[inl]]
        sigma = 0.01  # Estimated std.dev. of the lidar range measurements
        dim = 3 if absorient_domain == AbsorientDomain.SE2 else 6
        A = np.zeros((dim, dim))
        for i in range(p_inl.shape[0]):
            if absorient_domain == AbsorientDomain.SE2:
                Hi = np.array([[
                    normals[i, 0],
                    normals[i, 1],
                    normals[i, 1] * p_inl[i, 0] - normals[i, 0] * p_inl[i, 1],
                    ]])
            else:
                Hi = np.array([np.hstack((-normals[i, :], -np.cross(p_inl[i, :], normals[i, :])))])
            A += Hi.T @ Hi
        cov = np.power(sigma, 2) * np.linalg.inv(A + np.eye(dim) * 1e-15)
        if absorient_domain == AbsorientDomain.SE2:
            cov3 = cov
            cov = np.zeros((6, 6))
            cov[np.ix_((0, 1, 5), (0, 1, 5))] = cov3

    # Log ICP stats
    if iter >= max_iters / 2 or inl_errs and inl_errs[0] >= 0.01:
        log = logdebug if inl_errs[0] >= inl_errs[-1] else logwarn
        samples = sample(list(zip(inl_errs, inl_ratios))[::-1], 10)
        samples_str = '; '.join(['%.3g, %.2g' % ir for ir in samples])
        log('Inl. error, delta, ratio (from last): %s (%i it., %.2f s)', samples_str, iter, timer() - t)



    return IcpResult(T=T, num_iters=iter, correspondences=c, inliers=inl, p_inliers=p_inl, q_inliers=q_inl,
                     mean_inlier_dist=inl_errs[-1] if inl_errs else float('nan'), cov=cov)


def update_map(q_struct: Optional[np.ndarray], q_index: cKDTree, p_struct_registered: np.ndarray, min_dist: float,
               alignment: Alignment = Alignment.frame_to_frame) \
        -> Tuple[np.ndarray, cKDTree, int]:
    """Update the scan/map q according to the given update rule with new points from p_struct_registered.

    :param q_struct: Structured Numpy array with the last scan/current map q (may be None in the beginning).
                     Numpy array 3xN.
    :param q_index: KD-Tree index of the last scan/current map.
    :param p_struct_registered: The incoming cloud p aligned to the map frame.
    :param min_dist: Minimum distance between p points and their nearest neighbors to consider them new map points.
    :param alignment: Whether to align only to previous scan (frame_to_frame) or to current map (frame_to_map).

    :return New map, its KD-tree index and number of newly added points.
    """
    if q_struct is None or alignment == Alignment.frame_to_frame:
        q_struct = p_struct_registered
        num_new_points = np.ones(q_struct.shape, dtype=bool).sum()
    else:  # frame_to_map alignment with already existing map
        # TODO HW 3: Implement map update.
        # Look at the registered points from p_struct_registered and concatenate them to the old map q_struct such that
        # new points are not close to any old map point.
        p = position(p_struct_registered)
        q_struct = p_struct_registered  # FIXME q_struct should be the concatenation of new points and old map
        num_new_points = np.ones(q_struct.shape, dtype=bool).sum()  # FIXME this should be the number of added points


    q_index = cKDTree(position(q_struct))
    return q_struct, q_index, num_new_points


# TESTS #######################################################################

def affine(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = p2e(np.matmul(T, e2p(x, axis=0)), axis=0)
    return y


def rotation_z(angle: Union[float, np.ndarray], d=3) -> np.ndarray:
    T = np.eye(d + 1)
    cos = np.cos(angle)
    sin = np.sin(angle)
    T[:2, :2] = [[cos, -sin],
                 [sin,  cos]]
    return T


def translation(t: Union[np.ndarray, Sequence[float]], d=3) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t.ravel().tolist()
    T = np.eye(d + 1)
    T[:len(t), -1] = t
    return T


def points(n: int, low=0., high=1., d=3) -> np.ndarray:
    x = np.random.uniform(low, high, (d, n))
    return x


class TestAbsoluteOrientation(unittest.TestCase):

    def test_r3z45_2d(self):
        T_gt = rotation_z(np.pi / 4)
        x = points(10)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE2)
        self.assertTrue(np.allclose(T, T_gt))

    def test_r3z45(self):
        T_gt = rotation_z(np.pi / 4, d=3)
        x = points(10, d=3)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE3)
        self.assertTrue(np.allclose(T, T_gt))

    def test_r3z90_2d(self):
        T_gt = rotation_z(np.pi / 2)
        x = points(10)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE2)
        self.assertTrue(np.allclose(T, T_gt))

    def test_r3z90(self):
        T_gt = rotation_z(np.pi / 2, d=3)
        x = points(10, d=3)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE3)
        self.assertTrue(np.allclose(T, T_gt))

    def test_t3x_2d(self):
        T_gt = translation([1.])
        x = points(10)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE2)
        self.assertTrue(np.allclose(T, T_gt))

    def test_t3x(self):
        T_gt = translation([1.], d=3)
        x = points(10, d=3)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE3)
        self.assertTrue(np.allclose(T, T_gt))

    def test_t3y_2d(self):
        T_gt = translation([0., 1.])
        x = points(10)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE2)
        self.assertTrue(np.allclose(T, T_gt))

    def test_t3y(self):
        T_gt = translation([0., 1.], d=3)
        x = points(10, d=3)
        y = affine(T_gt, x)
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE3)
        self.assertTrue(np.allclose(T, T_gt))

    def test_negative_determinant_se3(self):
        # This leads to negative determinant of R at least in the teacher code.
        # The test verifies you are handling this case correctly.
        T_gt = np.array([
            [0.956108, 0.194712, 0.218960, -0.195608],
            [-0.168883, 0.976859, -0.131239, -0.145734],
            [-0.239447, 0.088500, 0.966867, -0.083456],
            [0.0, 0.0, 0.0, 1.0]
        ])
        x = np.array([
            [-0.360751390, 0.801239073, 1.151251196, -0.751251220, -1.006138205],
            [2.225036859, 1.298862934, 0.358819037, -0.158819049, 0.807106792],
            [0.198875606, -0.006014668, -0.067731253, 0.267731249, 0.312674701],
        ])
        y = np.array([
            [-0.099694535, 1.0, 1.0, -1.0, -1.0],
            [2.0, 1.023114442, 0.055058654, -0.213393121, 0.768085300],
            [0.378164887, -0.203290700, -0.424133777, 0.387260913, 0.567534267]
        ])
        T = absolute_orientation(x, y, domain=AbsorientDomain.SE3)
        self.assertTrue(np.allclose(T, T_gt))


class TestIcp(unittest.TestCase):

    def test_r3z(self):
        T_gt = rotation_z(np.radians(5.))
        x = points(100)
        x = unstructured_to_structured(x.T, names=['x', 'y', 'z'])
        y = transform(T_gt, x)
        T = icp(x, y).T
        self.assertTrue(np.allclose(T, T_gt))

    def test_t3x(self):
        T_gt = translation([0.001])
        x = points(100)
        x = unstructured_to_structured(x.T, names=['x', 'y', 'z'])
        y = transform(T_gt, x)
        T = icp(x, y).T
        self.assertTrue(np.allclose(T, T_gt, atol=1e-6))

    def test_plane_2d(self):
        T_gt = translation([0., 0., 0.])
        x = np.array([
            [-1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1],
            [0, 0, 0],
        ], dtype=np.float32)
        x = unstructured_to_structured(x.T, names=['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z'])
        y = x.copy()
        y['x'] += 1
        res = icp(x, y, loss=Loss.point_to_plane)
        self.assertTrue(np.allclose(res.T[:3, :3], T_gt[:3, :3], atol=1e-6))
        # Do not check x translation as it can be anything.
        self.assertTrue(np.allclose(res.T[1:, 3], T_gt[1:, 3], atol=1e-6))

    def test_plane_2d_p2point(self):
        T_gt = translation([0., 0., 0.])
        x = np.array([
            [-1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1],
            [0, 0, 0],
        ], dtype=np.float32)
        x = unstructured_to_structured(x.T, names=['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z'])
        y = x.copy()
        y['x'] += 1
        res = icp(x, y, loss=Loss.point_to_point, max_inlier_dist=0.5)
        self.assertTrue(np.allclose(res.T, T_gt, atol=1e-6))

    def test_plane_3d(self):
        T_gt = translation([0., 0., 0.])
        x = np.array([
            [-1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, -1, -1, -1],
            [0, 0, 0, 0],
        ], dtype=np.float32)
        x = unstructured_to_structured(x.T, names=['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z'])
        y = x.copy()
        y['x'] += 1
        y['x'][3] -= 0.1  # Without this, there would be two possible nearest neighbors for the last point
        res = icp(x, y, loss=Loss.point_to_plane, absorient_domain=AbsorientDomain.SE3)
        self.assertTrue(np.allclose(res.T[:3, :3], T_gt[:3, :3], atol=1e-6))
        # Do not check x translation as it can be anything.
        self.assertTrue(np.allclose(res.T[1:, 3], T_gt[1:, 3], atol=1e-6))

    def test_plane_3d_p2point(self):
        T_gt = translation([0., 0., 0.])
        x = np.array([
            [-1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [-1, -1, -1],
            [0, 0, 0],
        ], dtype=np.float32)
        x = unstructured_to_structured(x.T, names=['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z'])
        y = x.copy()
        y['x'] += 1
        res = icp(x, y, loss=Loss.point_to_point, absorient_domain=AbsorientDomain.SE3, max_inlier_dist=0.5)
        self.assertTrue(np.allclose(res.T, T_gt, atol=1e-6))


class TestUpdateMap(unittest.TestCase):

    def test_frame_to_frame(self):
        y = np.array([[0, 1,  0, -1],
                      [1, 0, -1,  0],
                      [0, 0,  0,  0]], dtype=np.float32)
        y_str = unstructured_to_structured(y.T, dtype=DType.position.value)
        x = y_str[:2]
        res = update_map(x, cKDTree(as_unstructured(x)), y_str[2:], 1.0, alignment=Alignment.frame_to_frame)
        new_map, _, num_new = res
        self.assertEquals(2, num_new)
        self.assertEquals((2,), new_map.shape)
        self.assertTrue(np.allclose(as_unstructured(y_str[2:]), as_unstructured(new_map)))

    def test_frame_to_map(self):
        y = np.array([[0, 1,  0, -1],
                      [1, 0, -1,  0],
                      [0, 0,  0,  0]], dtype=np.float32)
        y_str = unstructured_to_structured(y.T, dtype=DType.position.value)
        x = y_str[:2]
        res = update_map(x, cKDTree(as_unstructured(x)), y_str[2:], 1.0, alignment=Alignment.frame_to_map)
        new_map, _, num_new = res
        self.assertEquals(2, num_new)
        self.assertEquals((4,), new_map.shape)
        y_sorted = np.array(sorted(y.T.tolist()))
        new_sorted = np.array(sorted(as_unstructured(new_map).tolist()))
        self.assertTrue(np.allclose(y_sorted, new_sorted))


def main():
    np.random.seed(13)
    unittest.main(exit=False)


def absorient_demo():
    from aro_slam.utils import visualize_clouds_2d
    """
    A function to test implementation of Absolute Orientation algorithm.
    """
    # generate point clouds
    # initialize perturbation rotation
    theta_true = np.pi / 4
    R_true = np.eye(3)
    R_true[:2, :2] = np.array([[np.cos(theta_true), -np.sin(theta_true)], [np.sin(theta_true), np.cos(theta_true)]])
    t_true = np.array([[-2.0], [5.0], [0.0]])
    Tr_gt = np.eye(4)
    Tr_gt[:3, :3] = R_true
    Tr_gt[:3, 3:] = t_true

    # Generate data as a list of 2d points
    num_points = 30
    true_data = np.zeros((3, num_points))
    true_data[0, :] = range(0, num_points)
    true_data[1, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :])

    # Move the data
    moved_data = R_true.dot(true_data) + t_true

    # Add noise
    n = 0.5 * (np.random.random((3, num_points)) - 0.5)  # noise
    moved_data = moved_data + n

    # Assign to variables we use in formulas.
    Q = true_data
    P = moved_data

    # visualize not aligned point clouds
    visualize_clouds_2d(P.T, Q.T, markersize=4, title="Input clouds. Close the window to proceed.")

    # choose inliers for alignment
    n_inl = num_points
    # n_inl = 2
    inl_mask = np.random.choice(range(num_points), n_inl, replace=False)
    P_inl = P[:, inl_mask]
    Q_inl = Q[:, inl_mask]

    # run absolute orientation algorithm to estimate the transformation
    Tr = absolute_orientation(P_inl, Q_inl, domain=AbsorientDomain.SE2)
    Tr_inv = np.linalg.inv(Tr)
    theta = np.arctan2(Tr_inv[0, 0], Tr_inv[1, 0])
    theta_diff = np.abs(theta - theta_true)
    t_diff = np.linalg.norm(t_true.T - Tr_inv[:3, 3].T)

    print('ICP found transformation:\n%s\n' % (Tr,))
    print('GT transformation:\n%s\n' % (np.linalg.inv(Tr_gt),))
    print('Translation error: %.3f m, rotation error %.4f rad' % (t_diff, theta_diff))
    if t_diff < 0.1 and theta_diff < 0.01:
        print("absolute_orientation works OKAY")
    else:
        print("absolute_orientation works WRONG")

    # visualize the clouds after ICP alignment with found transformation
    P_aligned = Tr[:3, :3] @ P + Tr[:3, 3:]
    visualize_clouds_2d(P_aligned.T, Q.T, markersize=4, title="Aligned clouds. Close the window to proceed.")


def icp_demo():
    import os
    from aro_slam.utils import visualize_clouds_3d, filter_grid
    from aro_slam.icp_io import read_cloud, read_poses
    """
    The function utilizes a pair of point cloud scans captured in an indoor corridor-like environment.
    We run ICP algorithm to find a transformation that aligns the pair of clouds.
    The result is being tested along the ground-truth transformation available in a data set.
    The sample data is provided with the `aro_slam` package, however it is possible to use custom data, too.

    The script will automatically download data files to ~/.cache/fee_corridor .

    The `poses.csv` file contains 6DoF poses for all the point clouds from the data set as 4x4 matrices (T = [R|t])
    and corresponding indices of the scans which encodes recording time stamps in the format {sec}_{nsec}.
    For more information about the data, please, refer to https://paperswithcode.com/dataset/fee-corridor.
    """
    # download necessary data: 2 point clouds and poses
    # Use XDG_CACHE_HOME according to https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html .
    default_cache_dir = os.path.join(os.environ.get("HOME", "tmp"), ".cache")
    cache_dir = os.environ.get("XDG_CACHE_HOME", default_cache_dir)
    path = os.path.normpath(os.path.join(cache_dir, 'fee_corridor'))

    id1, id2 = '1669300991_618086656', '1669301026_319255296'

    if not os.path.exists(path):
        os.makedirs(path)
        url = 'http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2'
        os.system('wget %s/poses/poses.csv -P %s' % (url, path))
        os.system('wget %s/ouster_points/%s.npz -P %s' % (url, id1, path))
        os.system('wget %s/ouster_points/%s.npz -P %s' % (url, id2, path))

    # load cloud poses
    poses = read_poses(os.path.join(path, 'poses.csv'))
    pose1 = poses[id1]
    pose2 = poses[id2]

    # load point clouds
    cloud1 = read_cloud(os.path.join(path, '%s.npz' % id1))
    cloud2 = read_cloud(os.path.join(path, '%s.npz' % id2))

    # apply grid filtering to point clouds
    cloud1 = filter_grid(cloud1, grid_res=0.1)
    cloud2 = filter_grid(cloud2, grid_res=0.1)

    # visualize not aligned point clouds
    visualize_clouds_3d(cloud1, cloud2, markersize=0.3, title="Input clouds. Close the window to proceed.")

    # ground truth transformation that aligns the point clouds (from data set)
    Tr_gt = np.linalg.inv(pose2) @ pose1

    # run ICP algorithm to estimate the transformation (it is initialized with identity matrix)
    Tr_init = np.eye(4)
    loss = Loss.point_to_point
    res = icp(cloud1, cloud2, T0=Tr_init, inlier_ratio=0.9, inlier_dist_mult=2.0, max_iters=100,
              loss=loss, absorient_domain=AbsorientDomain.SE2)
    Tr_icp = res.T

    theta = np.arctan2(Tr_icp[0, 0], Tr_icp[1, 0])
    theta_true = np.arctan2(Tr_gt[0, 0], Tr_gt[1, 0])
    theta_diff = np.abs(theta - theta_true)
    t_diff = np.linalg.norm(Tr_gt[:3, 3].T - Tr_icp[:3, 3].T)

    print('ICP found transformation:\n%s\n' % Tr_icp)
    print('GT transformation:\n%s\n' % Tr_gt)
    print('ICP mean inliers distance: %.3f [m]' % res.mean_inlier_dist)
    print('ICP transform translation error: %.3f m, z rotation error %.4f rad' % (t_diff, theta_diff))
    if t_diff < (0.1 if loss == Loss.point_to_plane else 0.25) and theta_diff < 0.01:
        print("icp works OKAY")
    else:
        print("icp works WRONG")

    # visualize the clouds after ICP alignment with found transformation
    visualize_clouds_3d(transform(Tr_icp, cloud1), cloud2, markersize=0.3, title="Aligned clouds.")


if __name__ == '__main__':
    # main()
    absorient_demo()
    icp_demo()
