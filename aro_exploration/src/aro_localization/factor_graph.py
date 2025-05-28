"""
This code uses the following terminology from lectures:
- frame = coordinate system.
- wcf = World coordinate frame. The frame in which we localize the robot and markers. It doesn't change in time.
- rcf = Robot coordinate frame. The frame rigidly attached to the robot body. Position of rcf in wcf changes in time.
- yaw = $\\theta$. Heading (2D) rotation of the robot or of a frame. Usually expressed in radians.
- The factorgraph states x[t] are added at least once a second with the last received wheel-odometry measurement.
  The states are added more frequently when a marker is in view (in that case, corresponding odometry measurements
  are interpolated).
- In description of dimensions of various arrays, we use N to denote the number of measurements that have been added
  to the factorgraph (i.e. the number of successful calls to FactorGraph.add_z()). The estimated
  trajectory (FactorGraphState.x) is one element longer (it includes a starting pose), so its length is N+1.
- While variable indices t and timestamps are used interchangeably in the lectures for simplicity, they are not the same
  in reality. In this code, t denotes the index, e.g. state.x[t], while timestamps are expressed as nanoseconds since
  the beginning of time (in simulation, time starts at 0 every time the simulation is started; in reality, time starts
  on 1 Jan 1970, so current time is a bit above 1700000000 seconds). Although factorgraph states are added approximately
  once a second, it is not desired to make the simplification that variable indices are the same as timestamps. At least
  the marker observations come asynchronously and their exact time distance from the graph states is important.
  Timestamps are stored in fields like FactorGraphState.stamps. Timestamp of state.x[t] is state.stamps[t].

Documentation comments in this file use terminology 'Numpy array X' to denote numpy array with shape X.
So 'numpy array 3xN' means 2-dimensional numpy array with shape (3, N). (3 rows, N columns).
'numpy array 3' means 1-dimensional array with shape (3,). Please note that this sometimes creates grammatically weird
sentences.

Please, beware that (transpose of v) v.T == v for 1-dimensional vectors. To create a column vector from a 1-dimensional
vector, use col_vector() function.
"""

import contextlib
import copy
import io
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix

sys.path.append("../")
from aro_localization.least_squares import least_squares
from aro_localization.utils import (
    angular_signed_distance,
    atleast_2d_row,
    col_vector,
    coords_to_transform_matrix,
    forward,
    merge_dicts,
    normalize_angle_symmetric,
    nsec_to_sec,
    r2w,
    row_vector,
    transform_matrix_to_coords,
    w2r,
)

if __name__ == "__main__":
    logdebug = print
    loginfo = print

    def logwarn(*args, **kwargs):
        print(*args, **kwargs, file=sys.stderr)

    def logerr(*args, **kwargs):
        print(*args, **kwargs, file=sys.stderr)
else:
    try:
        import rospy

        logdebug = rospy.logdebug
        loginfo = rospy.loginfo
        logwarn = rospy.logwarn
        logerr = rospy.logerr
    except ImportError:
        import sys

        def logdebug(*args, **kwargs):
            pass

        loginfo = print

        def logwarn(*args, **kwargs):
            print(*args, **kwargs, file=sys.stderr)

        def logerr(*args, **kwargs):
            print(*args, **kwargs, file=sys.stderr)


class FactorGraphState:
    """Dynamically changing state of the factor graph."""

    def __init__(self):
        # Timestamps
        self.stamps: List[int] = []
        """Timestamps of factor graph states (int, nanoseconds). List N."""
        self.stamps_idx: Dict[int, int] = {}
        """Fast lookup index for stamps. Dict stamp->index in stamps."""

        self.last_optimized_idx = 0
        """The last index into self.x that has been optimized. All values after this index are just initial guesses."""

        # Trajectory estimate
        self.x = np.zeros((1, 3))
        """The best estimate of the robot trajectory in wcf. Elements are (x, y, yaw). Numpy array (N+1)x3."""
        self.x_odom = np.zeros((1, 3))
        """Trajectory of the robot constructed just by integrating the input odom measurements. Should be very similar
        to odom frame pose. Numpy array (N+1)x3."""
        self.x_icp_odom = np.zeros((1, 3))
        """Trajectory of the robot constructed just by integrating the input ICP odom measurements. Should be very
        similar to ICP odom frame pose. Numpy array (N+1)x3."""

        # Absolute marker
        self.z_ma = np.array([[np.nan, np.nan, np.nan]]).transpose()
        """Measurements of the absolute marker pose (in rcf[t]). Each time instant where no measurement is available
           should contain NaNs. Numpy array 3x(N+1)."""
        self.c_ma = np.zeros((1, 3, 3))
        """Cost matrices for the absolute marker pose residuals. Numpy array (N+1)x3x3."""
        self.seen_ma = False
        """Whether the absolute marker has already been seen."""

        # Relative marker
        self.mr = np.array(
            [[np.nan, np.nan, np.nan]],
        ).transpose()  # Initialization with NaNs
        """Current best estimate of the relative marker pose. Numpy array 3x1."""
        self.z_mr = np.array([[np.nan, np.nan, np.nan]]).transpose()
        """Measurements of the relative marker pose (in rcf[t]). Each time instant where no measurement is available
        should contain NaNs. Numpy array 3x(N+1)."""
        self.c_mr = np.zeros((1, 3, 3))
        """Cost matrices for the relative marker pose residuals. Numpy array (N+1)x3x3."""
        self.seen_mr = False
        """Whether the relative marker has already been seen."""

        # Odom
        self.z_odom = np.zeros((3, 0))
        """Odometry measurements (relative motion in rcf[t]). Each time instant contains a valid (non-NaN)
        measurement except a few last. Measurement at index t moves the robot from x[t] to x[t+1]. Numpy array 3xN."""
        self.c_odom = np.zeros((0, 3, 3))
        """Cost matrices for odometry motion residuals. Numpy array Nx3x3."""

        # ICP
        self.z_icp = np.zeros((3, 0))
        """Relative motion estimated from ICP (expressed in rcf). Each time instant where no measurement is available
           should contain NaNs. Measurement at index t moves the robot from
           x[t] to x[next t with non-NaN value in z_icp]. Numpy array 3xN."""
        self.c_icp = np.zeros((0, 3, 3))
        """Cost matrices for ICP motion residuals. Numpy array Nx3x3."""

        # Ground truth odometry
        self.z_gt_odom = np.zeros((3, 0))
        """Ground truth odometry measurements (absolute poses in wcf). Each time instant where no measurement is
        available should contain NaNs. Numpy array 3x(N+1)."""


class FactorGraphConfig:
    """Static configuration of the factor graph."""

    def __init__(
        self,
        ma=(12.0, 40.0, 0.0),
        mr_gt: Optional[np.ndarray] = None,
        fuse_icp=False,
        solver_options: Dict[str, Any] = None,
        method_options: Dict[str, Any] = None,
        icp_yaw_scale=1.0,
        marker_yaw_scale=1.0,
    ):
        """
        :param ma: Pose of the absolute marker in wcf (3-tuple x, y, yaw).
        :param mr_gt: Ground truth pose of the relative marker in wcf (3-tuple x, y, yaw).
        :param fuse_icp: Whether to fuse ICP SLAM odometry.
        :param solver_options: Options for the non-linear least squares solver. They will override the defaults.
        :param method_options: Options for the inner linear least squares solver. They will override the defaults.
        :param icp_yaw_scale: Additional scale applied to ICP odometry yaw residuals.
        :param marker_yaw_scale: Additional scale applied to marker yaw residuals.
        """
        # Absolute marker
        self.ma = col_vector(ma)
        """Pose of the absolute marker in wcf. Numpy array 3x1."""
        self.mr_gt = None if mr_gt is None else col_vector(mr_gt)
        """Ground truth pose of the relative marker in wcf (can be given as task input). Numpy array 3x1."""

        self.fuse_icp = fuse_icp
        """Whether to add ICP odometry measurements to the graph."""

        self.marker_yaw_scale = marker_yaw_scale
        self.icp_yaw_scale = icp_yaw_scale

        # Least-squares solver options.
        # Various (loss, f_scale) configs possible. See config/localization/solver.yaml for examples.
        # These values are used when running this file directly (executing the __main__ block). If it is run from ROS,
        # these values are overridden by whatever you put in config/localization/solver.yaml.
        # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html for details.
        self.solver_options = {
            "method": "cholmod",
            "x_scale": "jac",
            "loss": "linear",
            "f_scale": 1.0,
            "max_nfev": 40,
            "verbose": 1,
            "ftol": 5e-5,
            "gtol": 1e-8,
            "xtol": 5e-5,
        }
        self.all_methods_options = {
            "trf": {
                "maxiter": 10,
                "atol": 1e-4,
                "btol": 1e-4,
                "tr_solver": "lsmr",
            },
            "dogbox": {
                "maxiter": 10,
                "atol": 1e-4,
                "btol": 1e-4,
                "tr_solver": "lsmr",
            },
            "cholmod": {
                "beta": 1e-10,
                "use_AAt_decomposition": True,
                "ordering_method": "default",
            },
        }
        if solver_options is not None:
            merge_dicts(self.solver_options, solver_options)

        self.method_options = self.all_methods_options.get(
            self.solver_options["method"],
            {},
        )
        if method_options is not None:
            merge_dicts(self.method_options, method_options)

        loginfo("Using solver method: " + str(self.solver_options["method"]))
        loginfo("Solver options are: " + str(self.solver_options))
        loginfo("Method options are: " + str(self.method_options))

        self.solver_options["tr_options"] = self.method_options
        if "tr_solver" in self.method_options:
            self.solver_options["tr_solver"] = self.method_options["tr_solver"]


class FactorGraph:
    """
    Factorgraph implementation for 2D localization using relative motion measurements, absolute and relative
       markers and ICP odometry.

    Most methods of this class are static, handle a given pair of state and config and return a new state and config.
    This is mostly because of time synchronization and parallelization.
    """

    max_variable_dim = 3
    max_factors = 3333
    max_constraints = max_variable_dim * max_factors
    max_variables = max_variable_dim * max_factors + max_variable_dim
    max_factor_arity = 2

    _figure_shown = False

    def __init__(
        self,
        ma=(12.0, 40.0, 0.0),
        mr_gt: Optional[np.ndarray] = None,
        fuse_icp=False,
        solver_options: Optional[Dict[str, Any]] = None,
        method_options: Optional[Dict[str, Any]] = None,
        icp_yaw_scale=1.0,
        marker_yaw_scale=1.0,
    ):
        """
        :param ma: Pose of the absolute marker in wcf (3-tuple x, y, yaw).
        :param mr_gt: Ground truth pose of the relative marker in wcf (3-tuple x, y, yaw).
        :param fuse_icp: Whether to fuse ICP SLAM odometry.
        :param solver_options: Options for the non-linear least squares solver. They will override the defaults.
        :param method_options: Options for the inner linear least squares solver. They will override the defaults.
        :param icp_yaw_scale: Additional scale applied to ICP odometry yaw residuals.
        :param marker_yaw_scale: Additional scale applied to marker yaw residuals.
        """
        self.config = FactorGraphConfig(
            ma,
            mr_gt,
            fuse_icp,
            solver_options,
            method_options,
            icp_yaw_scale,
            marker_yaw_scale,
        )
        self.state = FactorGraphState()

    @staticmethod
    def add_z(
        state: FactorGraphState,
        config: FactorGraphConfig,
        odom_samples: Dict[int, Tuple[np.ndarray, np.ndarray]],
        z_mr: Sequence[float],
        c_mr: np.ndarray,
        z_ma: Sequence[float],
        c_ma: np.ndarray,
        icp_samples: Dict[int, Tuple[np.ndarray, np.ndarray]],
        z_gt_odom: Sequence[float],
        stamp: Optional[int] = None,
    ) -> FactorGraphState:
        """
        Add measurements to the factorgraph, append a new state.

        Costs of the measurements are computed based on the values configured in config/localization/costs_icp.yaml .
        Change values in the YAML file if you need different costs. The YAML config has no effect when this file is
        directly run (i.e. the __main__ section is executed).

        :param state: State of the factorgraph (will be changed by this method).
        :param config: Config of the factorgraph.
        :param odom_samples: Samples of wheeled odometry from a few seconds ago.
        :param z_mr: Measurement of the relative marker pose in robot body frame (NaNs if not observed). List 3.
        :param c_mr: Cost of the relative marker pose measurement residual. Numpy array 3x3.
        :param z_ma: Measurement of the absolute marker pose in robot body frame (NaNs if not observed). List 3.
        :param c_ma: Cost of the absolute marker pose measurement residual. Numpy array 3x3.
        :param icp_samples: Samples of ICP odometry from a few seconds ago.
        :param z_gt_odom: Measurement of ground truth absolute pose in world frame (NaNs if not observed). List 3.
        :param stamp: Timestamp of the measurement in nanoseconds.

        :return: The updated state.
        """
        if not state.seen_ma and np.all(np.isnan(z_ma)):
            logwarn("Waiting for absolute marker")
            return state
        if not state.seen_ma:
            # Initialize state.x[0] according to the first measurement of absolute marker. This aligns the coordinate
            # frame of the factorgraph with wcf.
            z_ma_in_rcf = coords_to_transform_matrix(z_ma)
            ma_in_wcf = coords_to_transform_matrix(config.ma)
            rcf_in_z_ma = np.linalg.inv(z_ma_in_rcf)
            rcf_in_wcf = ma_in_wcf @ rcf_in_z_ma  # Here, we assume z_ma_wcf == ma .
            state.x[0, :] = transform_matrix_to_coords(rcf_in_wcf)
            if np.any(np.isnan(z_gt_odom)):
                state.x_odom[0, :] = state.x[0, :]
                state.x_icp_odom[0, :] = state.x[0, :]
            else:
                state.x_odom[0, :] = z_gt_odom
                state.x_icp_odom[0, :] = z_gt_odom
            state.z_gt_odom = np.hstack((state.z_gt_odom, col_vector(z_gt_odom)))
            state.seen_ma = True
            state.z_ma[:, 0] = z_ma
            state.c_ma[0, :, :] = c_ma
            return state

        # Add odometry relative motion measurement
        state.stamps.append(stamp)
        state.stamps_idx[stamp] = len(state.stamps) - 1

        # Initialize new x with NaNs. It will be initialized from the best odometry source right before optimization.
        state.x = np.vstack((state.x, row_vector([np.nan] * state.x.shape[1])))

        state.z_gt_odom = np.hstack((state.z_gt_odom, col_vector(z_gt_odom)))
        # If there is no ground truth observation, just use the previous one (it should be close).
        if np.any(np.isnan(z_gt_odom)) and state.z_gt_odom.shape[1] > 1:
            state.z_gt_odom[:, -1] = state.z_gt_odom[:, -2]

        state.z_odom = np.hstack((state.z_odom, col_vector((np.nan,) * 3)))
        state.x_odom = np.vstack((state.x_odom, row_vector((np.nan,) * 3)))
        state.c_odom = np.vstack((state.c_odom, np.zeros((1, 3, 3))))
        for t in sorted(odom_samples.keys()):
            i = state.stamps_idx[t]
            state.z_odom[:, i], state.c_odom[i, :, :] = odom_samples[t]
            # Update self.x_odom by integrating odometry velocity
            state.x_odom[i + 1, :] = r2w(state.z_odom[:, i], state.x_odom[i, :]).ravel()

        # Add relative and absolute marker measurements
        state.z_ma = np.hstack((state.z_ma, col_vector(z_ma)))
        state.c_ma = np.vstack((state.c_ma, c_ma[None, :, :]))
        state.z_mr = np.hstack((state.z_mr, col_vector(z_mr)))
        state.c_mr = np.vstack((state.c_mr, c_mr[None, :, :]))
        # Put a gross weight on the first few abs. marker observations to make sure the FG will not move the start
        if state.z_ma.shape[1] < 10:
            state.c_ma[-1, :, :] *= 1e3

        if not np.isnan(z_mr[0]):
            state.seen_mr = True

        # Add ICP odometry measurement (add it even if self.fuse_icp is False - e.g. for visualization)
        state.z_icp = np.hstack((state.z_icp, col_vector((np.nan,) * 3)))
        state.c_icp = np.vstack((state.c_icp, np.zeros((1, 3, 3))))
        state.x_icp_odom = np.vstack(
            (state.x_icp_odom, row_vector([np.nan] * state.x_icp_odom.shape[1])),
        )
        for t in sorted(icp_samples.keys()):
            i = state.stamps_idx[t]
            state.z_icp[:, i], state.c_icp[i, :, :] = icp_samples[t]
            # Find last non-nan index in x_icp_odom. x_icp_odom[0] is non-nan, so the loop should always find an index.
            last_icp_t = i
            while np.any(np.isnan(state.x_icp_odom[last_icp_t, :])) and last_icp_t > 0:
                last_icp_t -= 1
            state.x_icp_odom[i + 1, :] = r2w(
                state.z_icp[:, i],
                state.x_icp_odom[last_icp_t, :],
            ).ravel()

        return state

    @staticmethod
    def optimize(
        state: FactorGraphState,
        config: FactorGraphConfig,
    ) -> FactorGraphState:
        """
        Optimize the given factor graph.

        :param state: State of the factorgraph (will be changed by this method).
        :param config: Config of the factorgraph.
        :return: New state with optimized `x` (robot trajectory estimate, Numpy array 3x(N+1)) and
                 `mr` (relative marker pose estimate, Numpy array 3x1).
        """
        x = state.x
        mr = state.mr
        z_odom = state.z_odom
        z_mr = state.z_mr
        z_ma = state.z_ma
        z_icp = state.z_icp

        # Indices telling which measurements are valid
        idx_odom = np.where((~np.isnan(z_odom)).all(axis=0))[0].astype(np.uint32)
        idx_mr = np.where((~np.isnan(z_mr)).all(axis=0))[0].astype(np.uint32)
        idx_ma = np.where((~np.isnan(z_ma)).all(axis=0))[0].astype(np.uint32)
        idx_icp = np.where((~np.isnan(z_icp)).all(axis=0))[0].astype(np.uint32)

        # States without odometry (and conditioned just by marker observations) can cause problems, but they offer a
        # faster state estimate.
        allow_states_without_odom = False
        if not allow_states_without_odom:
            idx_mr = np.intersect1d(idx_mr, idx_odom)
            idx_ma = np.intersect1d(idx_ma, idx_odom)

        # Initialize the not yet optimized part of the trajectory by either ICP or odom measurements
        for i in range(state.last_optimized_idx - 1, x.shape[0] - 1):
            if np.any(np.isnan(x[i + 1, :])):
                # Select the best available odometry source
                z = (
                    z_icp[:, i]
                    if i in idx_icp and not np.any(np.isnan(z_icp[:, i]))
                    else z_odom[:, i]
                )
                if np.any(np.isnan(z)):
                    # If there is no measurement, initialize with previous pose
                    x[i + 1, :] = x[i, :]
                else:
                    x[i + 1, :] = r2w(z, x[i, :])

        mr_is_nan = np.any(np.isnan(mr))
        if mr_is_nan:
            # If this is the first time the relative marker has been seen, set its pose estimate to the observed one
            if idx_mr.shape[0] > 0:
                mr = r2w(z_mr[:, idx_mr[0]], x[idx_mr[0], :])
            # If it has not been seen yet, set it to zeros because the optimizer refuses to work with NaNs
            else:
                mr = np.zeros_like(mr)

        x_ini = np.hstack((x.reshape(-1), mr.reshape(-1)))  # concatenation [x, mr]

        tic = time.perf_counter()
        logdebug(
            "Before opt: size: %i, x %s, mr %s, ma %s"
            % (
                state.last_optimized_idx,
                str(x[-1]),
                str(np.ravel(mr)),
                str(np.ravel(config.ma)),
            ),
        )

        solver_options = config.solver_options
        if x.shape[0] < 50:  # Give more time to the solver while the graph is small
            solver_options = copy.deepcopy(solver_options)
            solver_options["max_nfev"] *= 5

        with contextlib.redirect_stdout(io.StringIO()) as opt_log:
            sol: OptimizeResult = least_squares(
                FactorGraph.compute_residuals_opt,
                x_ini,
                FactorGraph.compute_residuals_jacobian_opt,
                args=(state, config, idx_odom, idx_mr, idx_ma, idx_icp),
                **solver_options,
            )
        for l in opt_log.getvalue().splitlines():
            if sol.success:
                logdebug("\t" + l)
            else:
                logerr("\t" + l)

        # "un-concatenate" x, mr
        x, mr = sol.x[0:-3].reshape(x.shape[0], x.shape[1]), sol.x[-3:].reshape(3, 1)
        x[:, 2] = normalize_angle_symmetric(
            x[:, 2],
        )  # Wrap the computed yaw between -pi and pi
        mr[2, :] = normalize_angle_symmetric(
            mr[2, :],
        )  # Wrap the computed marker yaw between -pi and pi

        # If c_odom is set to 0, it will probably happen that we will have nothing that would condition the last few
        # measurements in the optimization (icp odometry comes with delay, as well as markers). These last few
        # measurements would thus keep their initialization values. However, the optimization did probably move the
        # preceding parts of the trajectory somewhere else, so the unconditioned measurements would be connected to the
        # rest of the trajectory via a discrete jump. This is highly unwanted. So we look at the gradient of the
        # residuals, and if it is exactly zero, we know there was nothing conditioning the variables. So we just
        # recompute the tail of the trajectory so that it smoothly connects to the optimized part.
        grad = sol.grad[:-3].reshape(x.shape)
        for i in range(state.last_optimized_idx, x.shape[0]):
            if i > 0 and np.all(np.abs(grad[i, :]) == 0.0):
                if np.any(np.isnan(z_odom[:, i - 1])) or np.any(np.isnan(x[i - 1, :])):
                    break
                x[i, :] = atleast_2d_row(r2w(z_odom[:, i - 1], x[i - 1, :]))[:, 0]

        # If mr has not been seen yet, turn the result back to NaN
        if mr_is_nan and idx_mr.shape[0] == 0:
            mr *= np.nan

        state.last_optimized_idx = x.shape[0] - 1

        logdebug(
            "After opt: size %i, x %s, mr %s, ma %s"
            % (
                state.last_optimized_idx,
                str(x[-1]),
                str(np.ravel(mr)),
                str(np.ravel(config.ma)),
            ),
        )
        duration = time.perf_counter() - tic
        logdebug(
            "After opt: time %0.2f seconds, nfev %i, njev %i, "
            "time/it %0.3f s, time/fev %0.3f s, time/fev/size %0.2f ms"
            % (
                duration,
                sol.nfev,
                sol.njev,
                duration / sol.njev,
                duration / sol.nfev,
                (duration / sol.nfev) / state.last_optimized_idx * 1e3,
            ),
        )

        state.x = x
        state.mr = mr
        return state

    @staticmethod
    def res(
        x: np.ndarray,
        x2: np.ndarray,
        z: np.ndarray,
        cost: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the residual term for a relative pose measurement factor.
        It can be either marker observation or odometry.

        :param x: Estimated trajectory of the robot (Numpy array 3xN).
        :param x2: The absolute poses which correspond to the positions of the other end of the measurements (i.e.
                   estimated marker position or estimated position in the following time step). (Numpy array 3xN
                   or Numpy array 3).
        :param z: The relative measurements (i.e. marker pose in robot frame or the measured odometry).
                  (Numpy array 3xN).
        :param cost: The measurement costs (Numpy array Nx3x3).
        :return: Residuals for the relative pose measurement factor (Numpy array 3xN).
        """
        x2 = x2 if len(x2.shape) == 2 else x2[:, np.newaxis]

        # TODO HW 4: compute the residual of relative pose measurement z and difference of poses x2 and x
        # Basically, the residual is w2r(x2, x) - z . The application of cost to the residual is provided to make it
        # efficient. In the residual, you will want to use function angular_signed_distance() to compute differences
        # of angles.

        res_z = w2r(x2, x) - z

        res_z[2, :] = normalize_angle_symmetric(res_z[2, :])

        # The einsum() expression multiplies each column of res_z with a corresponding 3x3 submatrix from cost
        res_z = np.einsum("ijk,ji->ki", cost, res_z)  # apply cost

        return res_z

    @staticmethod
    def res_jac(
        x: np.ndarray,
        x2: np.ndarray,
        z: np.ndarray,
        cost: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Jacobian for one type of residuals given by res().

        :param x: Estimated positions of the robot (Numpy array 3xN).
        :param x2: The absolute poses which correspond to the positions of the other end of the measurement (i.e.
                   estimated marker positions or estimated positions in the following time step).
                   (Numpy array 3xN or 3).
        :param z: The relative measurements (i.e. marker poses in robot frame or the measured odometry).
                  (Numpy array 3xN).
        :param cost: The measurement costs (Numpy array Nx3x3).
        :return: Two Jacobians of the residuals - first w.r.t x and the second w.r.t x2. Two Numpy arrays 3x3xN.
        """
        x2 = x2 if len(x2.shape) == 2 else x2[:, np.newaxis]

        yaw = x[2, :]
        s = np.sin(yaw)
        c = np.cos(yaw)

        zeros = np.zeros_like(c)
        ones = np.ones_like(c)

        # TODO HW 4: compute the residual jacobian, derive res() w.r.t. x and x2 .
        # For efficiency, use the above precomputed vectors of yaw sine and cosine and vectors of ones or zeros.
        # Example usage to directly and efficiently create the 3x3xN matrix is (substitute the dots!):
        # J = np.array(( (., ., .), (., ., .), (zeros, zeros, ones) ))
        # This will create a stack of N 3x3 matrices, each of which has its last row equal to (0, 0, 1).
        xt = x[0, :]
        yt = x[1, :]
        xtpp = x2[0, :]
        ytpp = x2[1, :]
        J = np.array(
            (
                (-c, -s, -s * (xtpp - xt) + c * (ytpp - yt)),
                (s, -c, -c * (xtpp - xt) - s * (ytpp - yt)),
                (zeros, zeros, -ones),
            ),
        )
        J1 = np.array(
            ((c, s, zeros), (-s, c, zeros), (zeros, zeros, ones)),
        )

        # Multiply by cost matrices. The einsum basically takes the 3x3 cost matrix and the 3x3 Jacobian submatrix
        # and matrix-multiplies them, for each of the N items in the stack.
        J = np.einsum("lij,jkl->ikl", cost, J)
        J1 = np.einsum("lij,jkl->ikl", cost, J1)

        return J, J1

    @staticmethod
    def compute_residuals(
        state: FactorGraphState,
        config: FactorGraphConfig,
        idx_odom: np.ndarray,
        idx_mr: np.ndarray,
        idx_ma: np.ndarray,
        idx_icp: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute residuals (errors) of all measurements.

        :param state: State of the factorgraph (will not be changed by this method).
        :param config: Config of the factorgraph.
        :param idx_odom: Indices of z_odom with valid (non-NaN) measurements. Numpy array <N.
        :param idx_mr: Indices of z_mr with valid (non-NaN) measurements. Numpy array <N+1.
        :param idx_ma: Indices of z_ma with valid (non-NaN) measurements. Numpy array <N+1.
        :param idx_icp: Indices of z_icp with valid (non-NaN) measurements. Numpy array <N.
        :return: The residuals of odometry, relative marker, absolute marker and (possibly) ICP.
                 Tuple of numpy arrays Qx3, Q <= N.
        """
        x = state.x.transpose()

        # Odom measurements residuals
        x_next_odom = x[
            :,
            [i + 1 for i in idx_odom],
        ]  # States following the states with odometry measurements
        res_odom = FactorGraph.res(
            x[:, idx_odom],
            x_next_odom,
            state.z_odom[:, idx_odom],
            state.c_odom[idx_odom, :, :],
        )

        # Relative marker measurements residuals
        res_mr = FactorGraph.res(
            x[:, idx_mr],
            state.mr,
            state.z_mr[:, idx_mr],
            state.c_mr[idx_mr, :, :],
        )
        res_mr[2, :] *= config.marker_yaw_scale

        # Absolute marker measurements residuals
        res_ma = FactorGraph.res(
            x[:, idx_ma],
            config.ma,
            state.z_ma[:, idx_ma],
            state.c_ma[idx_ma, :, :],
        )
        res_ma[2, :] *= config.marker_yaw_scale

        res_icp = None
        if config.fuse_icp:
            # ICP odom measurements residuals
            x_next_icp = x[
                :,
                [i + 1 for i in idx_icp],
            ]  # States following the states with ICP measurements
            res_icp = FactorGraph.res(
                x[:, idx_icp],
                x_next_icp,
                state.z_icp[:, idx_icp],
                state.c_icp[idx_icp, :, :],
            )
            res_icp[2, :] *= config.icp_yaw_scale

        return res_odom, res_mr, res_ma, res_icp

    @staticmethod
    def compute_residuals_opt(
        xmr: np.ndarray,
        state: FactorGraphState,
        config: FactorGraphConfig,
        idx_odom: np.ndarray,
        idx_mr: np.ndarray,
        idx_ma: np.ndarray,
        idx_icp: np.ndarray,
    ) -> np.ndarray:
        """Like compute_residuals, but adapted shapes to work with the optimizer."""
        # unpack x, mr from the concatenated state
        state.x, state.mr = (
            xmr[0:-3].reshape(int((xmr.shape[0] - 3) / 3), 3, order="C"),
            xmr[-3:].reshape(3, 1),
        )

        res_odom, res_mr, res_ma, res_icp = FactorGraph.compute_residuals(
            state,
            config,
            idx_odom,
            idx_mr,
            idx_ma,
            idx_icp,
        )

        # Concatenate and linearize the residuals
        res = [
            res_odom.reshape(-1, order="F"),
            res_mr.reshape(-1, order="F"),
            res_ma.reshape(-1, order="F"),
        ]
        if config.fuse_icp:
            res.append(res_icp.reshape(-1, order="F"))

        return np.hstack(res)

    @staticmethod
    def compute_residuals_jacobian(
        state: FactorGraphState,
        config: FactorGraphConfig,
        idx_odom: np.ndarray,
        idx_mr: np.ndarray,
        idx_ma: np.ndarray,
        idx_icp: np.ndarray,
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
        """
        Compute Jacobians of the residuals. This is used to steer the optimization in a good direction.

        :param state: State of the factorgraph (will not be changed by this method).
        :param config: Config of the factorgraph.
        :param idx_odom: Indices of z_odom with valid (non-NaN) measurements. Numpy array <N.
        :param idx_mr: Indices of z_mr with valid (non-NaN) measurements. Numpy array <N+1.
        :param idx_ma: Indices of z_ma with valid (non-NaN) measurements. Numpy array <N+1.
        :param idx_icp: Indices of z_icp with valid (non-NaN) measurements. Numpy array <N.
        :return: 4-tuple of Jacobians of the residuals of odometry, relative marker, absolute marker and (possibly) ICP.
                 Tuple of sparse matrices 3Nx(3N+3), Q1x(3N+3), Q2x(3N+3), Q3x(3N+3), Q_ <= 3N+3.
        """
        x = state.x.transpose()

        num_positions, dim = x.shape[1], x.shape[0]
        num_rel_markers, num_abs_markers = state.mr.shape[1], config.ma.shape[1]
        num_variables = dim * (num_positions + num_rel_markers)

        assert dim == 3  # No other dimensions are supported in this library

        # Odom measurements
        z_odom = state.z_odom  # Odometry measurements
        c_odom = state.c_odom  # Costs of the odometry measurements
        x_next_odom = x[
            :,
            [i + 1 for i in idx_odom],
        ]  # States following the states with odometry measurements
        J, J1 = FactorGraph.res_jac(
            x[:, idx_odom],
            x_next_odom,
            z_odom[:, idx_odom],
            c_odom[idx_odom, :, :],
        )
        J_odom = FactorGraph.build_sparse_jacobian(J, J1, idx_odom, num_variables)

        # Relative marker measurements
        z_mr = state.z_mr  # Measurements of the relative marker
        c_mr = state.c_mr  # Costs of the relative marker measurements
        J, Jm = FactorGraph.res_jac(
            x[:, idx_mr],
            state.mr,
            z_mr[:, idx_mr],
            c_mr[idx_mr, :, :],
        )
        J[2, :, :] *= config.marker_yaw_scale
        Jm[2, :, :] *= config.marker_yaw_scale
        J_mr = FactorGraph.build_sparse_jacobian(
            J,
            Jm,
            idx_mr,
            num_variables,
            is_mr=True,
        )

        # Absolute marker measurements
        z_ma = state.z_ma  # Measurements of the absolute marker
        c_ma = state.c_ma  # Costs of the absolute marker measurements
        J, _ = FactorGraph.res_jac(
            x[:, idx_ma],
            config.ma,
            z_ma[:, idx_ma],
            c_ma[idx_ma, :, :],
        )
        J[2, :, :] *= config.marker_yaw_scale
        J_ma = FactorGraph.build_sparse_jacobian(J, None, idx_ma, num_variables)

        # ICP odom measurements
        z_icp = state.z_icp  # Measurements of the ICP odometry
        c_icp = state.c_icp  # Costs of the ICP odometry
        x_next_icp = x[
            :,
            [i + 1 for i in idx_icp],
        ]  # States following the states with ICP measurements
        J, J1 = FactorGraph.res_jac(
            x[:, idx_icp],
            x_next_icp,
            z_icp[:, idx_icp],
            c_icp[idx_icp, :, :],
        )
        J[2, :, :] *= config.icp_yaw_scale
        J1[2, :, :] *= config.icp_yaw_scale
        J_icp = FactorGraph.build_sparse_jacobian(J, J1, idx_icp, num_variables)

        return J_odom, J_mr, J_ma, J_icp

    @staticmethod
    def build_sparse_jacobian(
        J: np.ndarray,
        J1: Optional[np.ndarray],
        idxs: np.ndarray,
        num_variables: int,
        is_mr=False,
    ) -> csr_matrix:
        """
        Convert the given 3D stacks of Jacobians to the large 2D Jacobian, which is, moreover, sparse.

        :param J: Stack of Jacobians w.r.t. x[t]. Numpy array 3x3xN.
        :param J1: Stack of Jacobians w.r.t. x[t+1] or mr (see is_mr). Numpy array 3x3xN.
        :param idxs: Indices of valid measurements. Numpy int array N.
        :param num_variables: Total number of variables in the factor graph (number of columns of the Jacobian).
        :param is_mr: If True, interpret J1 as w.r.t mr. Otherwise, interpret it as w.r.t x[t+1].
        :return: Sparse matrix Nx{num_variables}.
        """
        dim = J.shape[0]
        factor_arity = 2 if J1 is not None else 1
        num_constraints = dim * J.shape[2]

        # This code looks complicated. It is written this way to achieve optimal efficiency building the sparse matrix
        # in the CSR format. A less efficient but equivalent algorithm would create a matrix which has the 3x3 blocks
        # from J along its main diagonal and the 3x3 blocks from J1 either next to J blocks (is_mr==False), or in the
        # rightmost column (is_mr==True).

        data = (
            np.hstack(
                (
                    np.moveaxis(J, -1, 0).reshape(
                        (J.shape[1] * J.shape[2], J.shape[0]),
                    ),
                    np.moveaxis(J1, -1, 0).reshape(
                        (J1.shape[1] * J1.shape[2], J1.shape[0]),
                    ),
                ),
            )
            if J1 is not None
            else np.moveaxis(J, -1, 0).reshape((J.shape[1] * J.shape[2], J.shape[0]))
        )

        indices = np.zeros((num_constraints, dim * factor_arity))
        indices[:, 0:dim] = np.repeat(
            np.repeat(idxs[np.newaxis, :] * dim, dim, axis=0).T,
            dim,
            axis=0,
        ) + np.arange(dim, dtype=np.uint32)
        if factor_arity > 1:
            if is_mr:
                indices[:, dim : (2 * dim)] = np.arange(
                    num_variables - dim,
                    num_variables,
                    dtype=np.uint32,
                )[np.newaxis, :]
            else:
                indices[:, dim : (2 * dim)] = indices[:, 0:dim] + dim

        indptr = np.arange(
            0,
            num_constraints * dim * factor_arity + 1,
            dim * factor_arity,
        )

        return csr_matrix(
            (data.ravel(), indices.ravel(), indptr),
            shape=(num_constraints, num_variables),
        )

    @staticmethod
    def compute_residuals_jacobian_opt(
        xmr: np.ndarray,
        state: FactorGraphState,
        config: FactorGraphConfig,
        idx_odom: np.ndarray,
        idx_mr: np.ndarray,
        idx_ma: np.ndarray,
        idx_icp: np.ndarray,
    ) -> csr_matrix:
        """Like compute_residuals_jacobian() but formatted for the optimizer, returns sparse matrix."""
        # unpack x, mr from the concatenated state
        state.x, state.mr = (
            xmr[0:-3].reshape(int((xmr.shape[0] - 3) / 3), 3, order="C"),
            xmr[-3:].reshape(3, 1),
        )

        J_odom, J_mr, J_ma, J_icp = FactorGraph.compute_residuals_jacobian(
            state,
            config,
            idx_odom,
            idx_mr,
            idx_ma,
            idx_icp,
        )

        J = (J_odom, J_mr, J_ma, J_icp) if config.fuse_icp else (J_odom, J_mr, J_ma)
        return scipy.sparse.vstack(J, format="csr")

    @staticmethod
    def visu(state: FactorGraphState, config: FactorGraphConfig, only_main=False):
        """Visualize the factorgraph inner workings."""
        x = state.x.transpose()
        x_odom = state.x_odom.transpose()
        x_icp_odom = state.x_icp_odom.transpose()
        mr = state.mr
        ma = config.ma
        z_odom = state.z_odom
        z_mr = state.z_mr
        z_ma = state.z_ma
        z_icp = state.z_icp
        z_gt_odom = state.z_gt_odom
        last_optimized_idx = state.last_optimized_idx

        # this happens right after the start before we have the first odom measurement
        if len(z_odom.shape) < 2 or z_odom.shape[1] == 0:
            return

        if x.shape[1] > z_odom.shape[1] + 1:
            x = x[:, : (z_odom.shape[1] + 1)]
            state.x = x.transpose()

        if last_optimized_idx is None:
            last_optimized_idx = x.shape[1] - 1

        idx_mr = np.where((~np.isnan(z_mr) & ~np.isnan(x)).all(axis=0))[0]
        idx_ma = np.where((~np.isnan(z_ma) & ~np.isnan(x)).all(axis=0))[0]
        idx_gt = np.where((~np.isnan(z_gt_odom) & ~np.isnan(x)).all(axis=0))[0]
        if (isinstance(z_gt_odom, np.ndarray) and z_gt_odom.shape[0] == 0) or (
            isinstance(z_gt_odom, list) and len(z_gt_odom) == 0
        ):
            idx_gt = np.array([])
        idx_x = np.where((~np.isnan(x)).all(axis=0))[0]
        idx_odom = np.where(
            (~np.isnan(z_odom) & ~np.isnan(x[:, :-1]) & ~np.isnan(x[:, 1:])).all(
                axis=0,
            ),
        )[0]
        idx_icp = np.where(
            (
                ~np.isnan(z_icp)
                & ~np.isnan(x[:, :-1])
                & ~np.isnan(x[:, 1:])
                & ~np.isnan(x_icp_odom[:, :-1])
            ).all(axis=0),
        )[0]

        z_mr_wcf = r2w(z_mr[:, idx_mr], x[:, idx_mr])
        z_ma_wcf = r2w(z_ma[:, idx_ma], x[:, idx_ma])
        z_icp_wcf = r2w(z_icp[:, idx_icp], x_icp_odom[:, idx_icp])
        gt_odom = np.array(z_gt_odom)

        nplots = 5  # update this when more subplots are added
        if idx_gt.shape[0] > 0:
            nplots += 2  # update this when more GT subplots are added
        if only_main:
            nplots = 2
        plot_index = 0

        fig = plt.figure(1, figsize=(10, 13))
        plt.clf()

        plot_index += 1
        main_ax: plt.Axes = plt.subplot(nplots, 1, (plot_index, plot_index + 1))
        plot_index += 1
        for k, j in enumerate(idx_ma):
            main_ax.plot(
                [x[0, j], z_ma_wcf[0, k]],
                [x[1, j], z_ma_wcf[1, k]],
                "x:",
                color="k",
                linewidth=1,
                mew=2,
            )
            main_ax.plot(
                [ma[0, 0], z_ma_wcf[0, k]],
                [ma[1, 0], z_ma_wcf[1, k]],
                ":",
                color="k",
                linewidth=1,
                mew=2,
            )

        for k, j in enumerate(idx_mr):
            main_ax.plot(
                [x[0, j], z_mr_wcf[0, k]],
                [x[1, j], z_mr_wcf[1, k]],
                "x:",
                color="c",
                linewidth=1,
                mew=2,
            )
            main_ax.plot(
                [mr[0, 0], z_mr_wcf[0, k]],
                [mr[1, 0], z_mr_wcf[1, k]],
                ":",
                color="c",
                linewidth=1,
                mew=2,
            )

        if idx_gt.shape[0] > 0:
            main_ax.plot(
                gt_odom[0, : (last_optimized_idx + 1)],
                gt_odom[1, : (last_optimized_idx + 1)],
                "-",
                color="r",
                linewidth=1,
                mew=2,
                label="GT odom",
            )
            main_ax.plot(
                gt_odom[0, last_optimized_idx:],
                gt_odom[1, last_optimized_idx:],
                "-",
                color=(1.0, 0.5, 0.5),
                linewidth=1,
                mew=2,
            )

        for k, j in enumerate(idx_icp):
            main_ax.plot(
                [x_icp_odom[0, j], z_icp_wcf[0, k]],
                [x_icp_odom[1, j], z_icp_wcf[1, k]],
                ":",
                color=(0.6, 0.4, 0.5),
                linewidth=2,
                mew=2,
            )

        main_ax.plot(
            x_odom[0, idx_odom[idx_odom <= last_optimized_idx]],
            x_odom[1, idx_odom[idx_odom <= last_optimized_idx]],
            "-",
            color="y",
            linewidth=1,
            mew=2,
            label="Integrated odom",
        )
        main_ax.plot(
            x_odom[0, idx_odom[idx_odom >= last_optimized_idx]],
            x_odom[1, idx_odom[idx_odom >= last_optimized_idx]],
            "-",
            color=(1.0, 1.0, 0.5),
            linewidth=1,
            mew=2,
        )
        main_ax.plot(
            x_icp_odom[0, idx_icp[idx_icp <= last_optimized_idx]],
            x_icp_odom[1, idx_icp[idx_icp <= last_optimized_idx]],
            "-",
            color="g",
            linewidth=1,
            mew=2,
            label="Integrated ICP odom",
        )
        main_ax.plot(
            x_icp_odom[0, idx_icp[idx_icp >= last_optimized_idx]],
            x_icp_odom[1, idx_icp[idx_icp >= last_optimized_idx]],
            "-",
            color=(0.5, 1.0, 0.5),
            linewidth=1,
            mew=2,
        )
        main_ax.plot(
            x[0, idx_x[idx_x <= last_optimized_idx]],
            x[1, idx_x[idx_x <= last_optimized_idx]],
            "-",
            color="b",
            linewidth=2,
            mew=2,
            label="Fused odom",
        )
        main_ax.plot(
            x[0, idx_x[idx_x >= last_optimized_idx]],
            x[1, idx_x[idx_x >= last_optimized_idx]],
            "-",
            color=(0.5, 0.5, 1.0),
            linewidth=2,
            mew=2,
        )
        if len(idx_odom) > 0:
            main_ax.scatter(
                x_odom[0, idx_odom[-1]],
                x_odom[1, idx_odom[-1]],
                color="y",
                s=40,
            )
        if len(idx_icp) > 0:
            main_ax.scatter(
                x_icp_odom[0, idx_icp[-1]],
                x_icp_odom[1, idx_icp[-1]],
                color="g",
                s=40,
            )
        if len(idx_x) > 0:
            main_ax.scatter(x[0, idx_x[-1]], x[1, idx_x[-1]], color="b", s=80)
        if idx_gt.shape[0] > 0:
            main_ax.scatter(gt_odom[0, -1], gt_odom[1, -1], color="r", s=40)

        if config.mr_gt is not None:
            main_ax.plot(
                config.mr_gt[0, 0],
                config.mr_gt[1, 0],
                "s",
                color="y",
                linewidth=1,
                mew=6,
                label="Rel.m. GT",
            )
        if not np.any(np.isnan(mr)):
            main_ax.plot(
                mr[0, :],
                mr[1, :],
                "s",
                color="c",
                linewidth=1,
                mew=6,
                label="Rel. marker",
            )
        else:
            main_ax.plot(
                [0],
                [0],
                "s",
                color="c",
                linewidth=1,
                mew=6,
                label="Rel. marker",
            )
        main_ax.plot(
            ma[0, :],
            ma[1, :],
            "s",
            color="k",
            linewidth=1,
            mew=6,
            label="Abs. marker",
        )

        main_ax.axis("equal")
        main_ax.set_xlabel("x")
        main_ax.set_ylabel("y")
        # main_ax.set_xlim(-2.5, 2.5)
        # main_ax.set_ylim(-2.5, 2.5)
        main_ax.grid()
        main_ax.legend()

        if only_main:
            return

        stamps = nsec_to_sec(
            np.array(state.stamps + [state.stamps[-1]]) - state.stamps[0],
        )

        plot_index += 1
        yaw_ax: plt.Axes = plt.subplot(nplots, 1, plot_index)

        def safe_plt(obj):
            old = obj.plot

            def yaw_plt(a, b, *args, **kwargs):
                min_len = min(len(a), len(b))
                old(a[:min_len], b[:min_len], *args, **kwargs)

            return yaw_plt

        yaw_ax.plot = safe_plt(yaw_ax)
        if idx_gt.shape[0] > 0:
            yaw_ax.plot(
                stamps[idx_gt[idx_gt <= last_optimized_idx]],
                gt_odom[2, idx_gt[idx_gt <= last_optimized_idx]],
                "-",
                color="r",
                linewidth=1,
                mew=2,
                label="GT odom",
            )
            yaw_ax.plot(
                stamps[idx_gt[idx_gt >= last_optimized_idx]],
                gt_odom[2, idx_gt[idx_gt >= last_optimized_idx]],
                "-",
                color=(1.0, 0.5, 0.5),
                linewidth=1,
                mew=2,
            )
        yaw_ax.plot(
            stamps[idx_odom[idx_odom <= last_optimized_idx]],
            x_odom[2, idx_odom[idx_odom <= last_optimized_idx]],
            "-",
            color="y",
            linewidth=1,
            mew=2,
            label="Integrated odom",
        )
        yaw_ax.plot(
            stamps[idx_odom[idx_odom >= last_optimized_idx]],
            x_odom[2, idx_odom[idx_odom >= last_optimized_idx]],
            "-",
            color=(1.0, 1.0, 0.5),
            linewidth=1,
            mew=2,
        )
        yaw_ax.plot(
            stamps[idx_icp[idx_icp <= last_optimized_idx]],
            x_icp_odom[2, idx_icp[idx_icp <= last_optimized_idx]],
            "-",
            color="g",
            linewidth=1,
            mew=2,
            label="Integrated ICP odom",
        )
        yaw_ax.plot(
            stamps[idx_icp[idx_icp >= last_optimized_idx]],
            x_icp_odom[2, idx_icp[idx_icp >= last_optimized_idx]],
            "-",
            color=(0.5, 1.0, 0.5),
            linewidth=1,
            mew=2,
        )
        yaw_ax.plot(
            stamps[idx_x[idx_x <= last_optimized_idx]],
            x[2, idx_x[idx_x <= last_optimized_idx]],
            "-",
            color="b",
            linewidth=1,
            mew=2,
            label="Fused odom",
        )
        yaw_ax.plot(
            stamps[idx_x[idx_x >= last_optimized_idx]],
            x[2, idx_x[idx_x >= last_optimized_idx]],
            "-",
            color=(0.5, 0.5, 1.0),
            linewidth=1,
            mew=2,
        )

        yaw_ax.set_ylabel("yaw")
        yaw_ax.set_yticks(np.arange(-1.5 * np.pi, 1.5 * np.pi, np.pi / 2))
        yaw_ax.grid()

        plot_index += 1
        res_ax: plt.Axes = plt.subplot(nplots, 1, plot_index)

        res_scaled_odom, res_scaled_mr, res_scaled_ma, res_scaled_icp = (
            FactorGraph.compute_residuals(
                state,
                config,
                idx_odom,
                idx_mr,
                idx_ma,
                idx_icp,
            )
        )

        state.c_odom[:] = np.repeat(
            np.eye(3)[None, :, :],
            state.c_odom.shape[0],
            axis=0,
        )
        state.c_mr[:] = np.repeat(np.eye(3)[None, :, :], state.c_mr.shape[0], axis=0)
        state.c_ma[:] = np.repeat(np.eye(3)[None, :, :], state.c_ma.shape[0], axis=0)
        state.c_icp[:] = np.repeat(np.eye(3)[None, :, :], state.c_icp.shape[0], axis=0)
        res_odom, res_mr, res_ma, res_icp = FactorGraph.compute_residuals(
            state,
            config,
            idx_odom,
            idx_mr,
            idx_ma,
            idx_icp,
        )

        res_ax.plot = safe_plt(res_ax)
        res_ax.plot(
            stamps[idx_odom[idx_odom <= last_optimized_idx]],
            np.linalg.norm(res_odom, axis=0)[idx_odom <= last_optimized_idx],
            label="res_odom",
            color="k",
        )
        res_ax.plot(
            stamps[idx_odom[idx_odom >= last_optimized_idx]],
            np.linalg.norm(res_odom, axis=0)[idx_odom >= last_optimized_idx],
            color=(0.5, 0.5, 0.5),
        )
        res_ax.plot(
            stamps[idx_mr[idx_mr <= last_optimized_idx]],
            np.linalg.norm(res_mr, axis=0)[idx_mr <= last_optimized_idx],
            label="res_mr",
            color="r",
        )
        res_ax.plot(
            stamps[idx_mr[idx_mr >= last_optimized_idx]],
            np.linalg.norm(res_mr, axis=0)[idx_mr >= last_optimized_idx],
            color=(1.0, 0.5, 0.5),
        )
        res_ax.plot(
            stamps[idx_ma[idx_ma <= last_optimized_idx]],
            np.linalg.norm(res_ma, axis=0)[idx_ma <= last_optimized_idx],
            label="res_ma",
            color="g",
        )
        res_ax.plot(
            stamps[idx_ma[idx_ma >= last_optimized_idx]],
            np.linalg.norm(res_ma, axis=0)[idx_ma >= last_optimized_idx],
            color=(0.5, 1.0, 0.5),
        )
        if config.fuse_icp:
            res_ax.plot(
                stamps[idx_icp[idx_icp <= last_optimized_idx]],
                np.linalg.norm(res_icp, axis=0)[idx_icp <= last_optimized_idx],
                label="res_icp",
                color="b",
            )
            res_ax.plot(
                stamps[idx_icp[idx_icp >= last_optimized_idx]],
                np.linalg.norm(res_icp, axis=0)[idx_icp >= last_optimized_idx],
                color=(0.5, 0.5, 1.0),
            )

        res_ax.set_ylabel("unscaled residuals")
        res_ax.grid()
        res_ax.legend(loc="upper left", frameon=False)

        plot_index += 1
        res_scaled_ax: plt.Axes = plt.subplot(nplots, 1, plot_index)

        res_scaled_ax.plot = safe_plt(res_scaled_ax)
        res_scaled_ax.plot(
            stamps[idx_odom[idx_odom <= last_optimized_idx]],
            np.linalg.norm(res_scaled_odom, axis=0)[idx_odom <= last_optimized_idx],
            label="res_scaled_odom",
            color="k",
        )
        res_scaled_ax.plot(
            stamps[idx_odom[idx_odom >= last_optimized_idx]],
            np.linalg.norm(res_scaled_odom, axis=0)[idx_odom >= last_optimized_idx],
            color=(0.5, 0.5, 0.5),
        )
        res_scaled_ax.plot(
            stamps[idx_mr[idx_mr <= last_optimized_idx]],
            np.linalg.norm(res_scaled_mr, axis=0)[idx_mr <= last_optimized_idx],
            label="res_scaled_mr",
            color="r",
        )
        res_scaled_ax.plot(
            stamps[idx_mr[idx_mr >= last_optimized_idx]],
            np.linalg.norm(res_scaled_mr, axis=0)[idx_mr >= last_optimized_idx],
            color=(1.0, 0.5, 0.5),
        )
        res_scaled_ax.plot(
            stamps[idx_ma[idx_ma <= last_optimized_idx]],
            np.linalg.norm(res_scaled_ma, axis=0)[idx_ma <= last_optimized_idx],
            label="res_scaled_ma",
            color="g",
        )
        res_scaled_ax.plot(
            stamps[idx_ma[idx_ma >= last_optimized_idx]],
            np.linalg.norm(res_scaled_ma, axis=0)[idx_ma >= last_optimized_idx],
            color=(0.5, 1.0, 0.5),
        )
        if config.fuse_icp:
            res_scaled_ax.plot(
                stamps[idx_icp[idx_icp <= last_optimized_idx]],
                np.linalg.norm(res_scaled_icp, axis=0)[idx_icp <= last_optimized_idx],
                label="res_scaled_icp",
                color="b",
            )
            res_scaled_ax.plot(
                stamps[idx_icp[idx_icp >= last_optimized_idx]],
                np.linalg.norm(res_scaled_icp, axis=0)[idx_icp >= last_optimized_idx],
                color=(0.5, 0.5, 1.0),
            )

        if idx_gt.shape[0] == 0:
            res_scaled_ax.set_xlabel("t")
        res_scaled_ax.set_ylabel("scaled residuals")
        res_scaled_ax.grid()
        res_scaled_ax.legend(loc="upper left", frameon=False)

        if idx_gt.shape[0] > 0:
            plot_index += 1
            gt_ax: plt.Axes = plt.subplot(nplots, 1, plot_index)
            errors_x = x[0, idx_x] - gt_odom[0, idx_x]
            errors_y = x[1, idx_x] - gt_odom[1, idx_x]
            errors_yaw = angular_signed_distance(x[2, idx_x], gt_odom[2, idx_x])
            errors = np.linalg.norm(np.vstack((errors_x, errors_y, errors_yaw)), axis=0)
            gt_ax.plot = safe_plt(gt_ax)
            gt_ax.plot(
                stamps[idx_x[idx_x <= last_optimized_idx]],
                errors[idx_x[idx_x <= last_optimized_idx]],
                label="FG localization error",
                color="k",
            )
            gt_ax.plot(
                stamps[idx_x[idx_x >= last_optimized_idx]],
                errors[idx_x[idx_x >= last_optimized_idx]],
                color=(0.5, 0.5, 0.5),
            )
            gt_ax.plot(
                stamps[idx_x[idx_x <= last_optimized_idx]],
                errors_x[idx_x[idx_x <= last_optimized_idx]],
                label="X error",
                color="r",
            )
            gt_ax.plot(
                stamps[idx_x[idx_x >= last_optimized_idx]],
                errors_x[idx_x[idx_x >= last_optimized_idx]],
                color=(1.0, 0.5, 0.5),
            )
            gt_ax.plot(
                stamps[idx_x[idx_x <= last_optimized_idx]],
                errors_y[idx_x[idx_x <= last_optimized_idx]],
                label="Y error",
                color="g",
            )
            gt_ax.plot(
                stamps[idx_x[idx_x >= last_optimized_idx]],
                errors_y[idx_x[idx_x >= last_optimized_idx]],
                color=(0.5, 1.0, 0.5),
            )
            gt_ax.plot(
                stamps[idx_x[idx_x <= last_optimized_idx]],
                errors_yaw[idx_x[idx_x <= last_optimized_idx]],
                label="Yaw error",
                color="b",
            )
            gt_ax.plot(
                stamps[idx_x[idx_x >= last_optimized_idx]],
                errors_yaw[idx_x[idx_x >= last_optimized_idx]],
                color=(0.5, 0.5, 1.0),
            )
            gt_ax.set_ylabel("fused error")
            ylim_min = (
                np.nanmin(np.hstack((errors_x, errors_y, errors_yaw, [-1]))) * 1.1
            )
            ylim_max = (
                np.nanmax(np.hstack((errors_x, errors_y, errors_yaw, errors, [-1])))
                * 1.1
            )
            gt_ax.set_ylim(ylim_min, ylim_max)
            gt_ax.grid()
            gt_ax.legend(loc="upper left", frameon=False)

            plot_index += 1
            gt_odom_ax: plt.Axes = plt.subplot(nplots, 1, plot_index)
            x_odom2 = x_odom[:, : gt_odom.shape[1]]
            odom_and_gt_idxs = np.where(
                (~np.isnan(x_odom2) & ~np.isnan(gt_odom)).all(axis=0),
            )[0]
            if odom_and_gt_idxs.shape[0] > 0:
                odom_errors_x = (
                    x_odom2[0, odom_and_gt_idxs] - gt_odom[0, odom_and_gt_idxs]
                )
                odom_errors_y = (
                    x_odom2[1, odom_and_gt_idxs] - gt_odom[1, odom_and_gt_idxs]
                )
                odom_errors_yaw = angular_signed_distance(
                    x_odom2[2, odom_and_gt_idxs],
                    gt_odom[2, odom_and_gt_idxs],
                )
                odom_errors = np.linalg.norm(
                    np.vstack((odom_errors_x, odom_errors_y, odom_errors_yaw)),
                    axis=0,
                )
                gt_odom_ax.plot = safe_plt(gt_odom_ax)
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]],
                    odom_errors[
                        odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]
                    ],
                    label="Odom localization error",
                    color="m",
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]],
                    odom_errors[
                        odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]
                    ],
                    color=(0.5, 1.0, 1.0),
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]],
                    odom_errors_x[
                        odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]
                    ],
                    label="X error",
                    color="r",
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]],
                    odom_errors_x[
                        odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]
                    ],
                    color=(1.0, 0.5, 0.5),
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]],
                    odom_errors_y[
                        odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]
                    ],
                    label="Y error",
                    color="g",
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]],
                    odom_errors_y[
                        odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]
                    ],
                    color=(0.5, 1.0, 0.5),
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]],
                    odom_errors_yaw[
                        odom_and_gt_idxs[odom_and_gt_idxs <= last_optimized_idx]
                    ],
                    label="Yaw error",
                    color="b",
                )
                gt_odom_ax.plot(
                    stamps[odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]],
                    odom_errors_yaw[
                        odom_and_gt_idxs[odom_and_gt_idxs >= last_optimized_idx]
                    ],
                    color=(0.5, 0.5, 1.0),
                )
                ylim_min = (
                    np.nanmin(
                        np.hstack(
                            (odom_errors_x, odom_errors_y, odom_errors_yaw, [-1]),
                        ),
                    )
                    * 1.1
                )
                ylim_max = (
                    np.nanmax(
                        np.hstack(
                            (
                                odom_errors_x,
                                odom_errors_y,
                                odom_errors_yaw,
                                odom_errors,
                                [-1],
                            ),
                        ),
                    )
                    * 1.1
                )
                gt_odom_ax.set_ylim(ylim_min, ylim_max)
            gt_odom_ax.set_xlabel("t")
            gt_odom_ax.set_ylabel("odom error")
            gt_odom_ax.grid()
            gt_odom_ax.legend(loc="upper left", frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.01)
        plt.savefig("/tmp/fig%04i.png" % (x.shape[1],))

        # plt.pause() brings the plot window into foreground on each redraw, which is not convenient, so we only call it
        # the first time and then we use a workaround that hopefully works on all plotting backends (but not sure!)
        if not FactorGraph._figure_shown:
            plt.pause(0.001)
            FactorGraph._figure_shown = True
        else:
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)


class TestFactorGraph(unittest.TestCase):
    def generate_test_data(self):
        config = FactorGraphConfig()
        config.ma = col_vector([1.0, 0.0, 0.0])
        config.fuse_icp = True

        state = FactorGraphState()
        state.mr = col_vector([2.0, 0.0, 0.0])
        state.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        state.z_odom = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).transpose()
        state.z_mr = np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).transpose()
        state.z_ma = np.array([[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]).transpose()
        state.z_icp = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).transpose()
        state.c_odom = np.array([np.eye(3), np.eye(3)])
        state.c_mr = np.array([np.eye(3), np.eye(3)])
        state.c_ma = np.array([np.eye(3), np.eye(3)])
        state.c_icp = np.array([np.eye(3), np.eye(3)])
        state.stamps = [1, 2]
        state.stamps_idx = {1: 0, 2: 1}
        state.t_markers = [1, 2]

        idx_odom = np.arange(2)
        idx_mr = np.arange(2)
        idx_ma = np.arange(1)
        idx_icp = np.arange(2)

        return config, state, idx_odom, idx_mr, idx_ma, idx_icp

    def test_res(self):
        config, state, idx_odom, idx_mr, idx_ma, idx_icp = self.generate_test_data()

        res = FactorGraph.res(
            state.x[:-1, :].T,
            state.x[1:, :].T,
            state.z_odom,
            state.c_odom,
        )

        self.assertSequenceEqual(res.shape, (3, 2))
        self.assertSequenceEqual(
            res.ravel().tolist(),
            np.zeros((3, 2)).ravel().tolist(),
        )

    def test_res_jac(self):
        config, state, idx_odom, idx_mr, idx_ma, idx_icp = self.generate_test_data()

        J1, J2 = FactorGraph.res_jac(
            state.x[:-1, :].T,
            state.x[1:, :].T,
            state.z_odom,
            state.c_odom,
        )

        self.assertSequenceEqual(J1.shape, (3, 3, 2))
        self.assertSequenceEqual(J2.shape, (3, 3, 2))
        self.assertSequenceEqual(
            J1[:, :, 0].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J2[:, :, 0].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J1[:, :, 1].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J2[:, :, 1].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )

    def test_compute_residuals(self):
        config, state, idx_odom, idx_mr, idx_ma, idx_icp = self.generate_test_data()

        res_odom, res_mr, res_ma, res_icp = FactorGraph.compute_residuals(
            state,
            config,
            idx_odom,
            idx_mr,
            idx_ma,
            idx_icp,
        )

        self.assertSequenceEqual(
            res_odom.ravel().tolist(),
            np.zeros((3, 2)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            res_mr.ravel().tolist(),
            np.zeros((3, 2)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            res_ma.ravel().tolist(),
            np.zeros((3, 1)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            res_icp.ravel().tolist(),
            np.zeros((3, 2)).ravel().tolist(),
        )

    def test_compute_residuals_jacobian(self):
        config, state, idx_odom, idx_mr, idx_ma, idx_icp = self.generate_test_data()

        J_odom, J_mr, J_ma, J_icp = FactorGraph.compute_residuals_jacobian(
            state,
            config,
            idx_odom,
            idx_mr,
            idx_ma,
            idx_icp,
        )
        J_odom = J_odom.toarray()
        J_mr = J_mr.toarray()
        J_ma = J_ma.toarray()
        J_icp = J_icp.toarray()

        self.assertSequenceEqual(J_odom.shape, (6, 12))
        self.assertSequenceEqual(J_mr.shape, (6, 12))
        self.assertSequenceEqual(J_ma.shape, (3, 12))
        self.assertSequenceEqual(J_icp.shape, (6, 12))
        self.assertSequenceEqual(
            J_odom[:3, :3].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_odom[3:, :3].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_odom[:3, 3:6].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_odom[3:, 3:6].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_odom[:3, 6:9].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_odom[3:, 6:9].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_odom[:3, 9:12].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_odom[3:, 9:12].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )

        self.assertSequenceEqual(
            J_mr[:3, :3].ravel().tolist(),
            [-1, 0, 0, 0, -1, -2, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_mr[3:, :3].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_mr[:3, 3:6].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_mr[3:, 3:6].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_mr[:3, 6:9].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_mr[3:, 6:9].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_mr[:3, 9:12].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_mr[3:, 9:12].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )

        self.assertSequenceEqual(
            J_ma[:3, :3].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_ma[:3, 3:6].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_ma[:3, 6:9].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_ma[:3, 9:12].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )

        self.assertSequenceEqual(
            J_icp[:3, :3].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_icp[3:, :3].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_icp[:3, 3:6].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_icp[3:, 3:6].ravel().tolist(),
            [-1, 0, 0, 0, -1, -1, 0, 0, -1],
        )
        self.assertSequenceEqual(
            J_icp[:3, 6:9].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_icp[3:, 6:9].ravel().tolist(),
            np.eye(3).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_icp[:3, 9:12].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )
        self.assertSequenceEqual(
            J_icp[3:, 9:12].ravel().tolist(),
            np.zeros((3, 3)).ravel().tolist(),
        )


def main():
    test = unittest.main(exit=False)
    if not test.result.wasSuccessful():
        sys.exit(1)


# This code is for playing with the optimization of factorgraph. It generates a sample trajectory and runs the
# factorgraph localization on it.


def sim(
    trajectory_length=10,
    v=0.3,
    w=0.1,
    x0=np.array([0.0, 0.0, 0.0]),
    mr=np.array([[2.0], [1.0], [0.0]]),
    ma=np.array([[1.0], [0.0], [0.0]]),
    noise_odom=0.1,
    noise_mr=0.1,
    noise_ma=0.1,
    noise_icp=0.1,
):
    """
    Generate a spiral trajectory with the given parameters.

    :param int trajectory_length: Number of steps of the trajectory.
    :param float v: Linear velocity.
    :param float w: Angular velocity.
    :param np.ndarray x0: Initial pose estimate. Numpy array 3.
    :param np.ndarray mr: Relative marker pose. Numpy array 3x1.
    :param np.ndarray ma: Absolute marker pose. Numpy array 3x1.
    :param float noise_odom: Noise of odometry measurements.
    :param float noise_mr: Noise of relative marker measurements.
    :param float noise_ma: Noise of absolute marker measurements.
    :param float noise_icp: Noise of ICP odometry measurements.
    :return: x0, x, mr, ma, z_odom, z_ma, z_mr, z_icp
    """
    np.set_printoptions(formatter={"float_kind": "{:.4f}".format})

    # ground truth robot positions x and marker mr
    x0 = x0.reshape(3, 1)  # initial position
    u = np.ones((3, trajectory_length - 1), dtype=float)
    u[0, :] = u[0, :] * v
    u[1, :] = 0
    u[2, :] = u[2, :] * w
    x = forward(
        u,
        x0,
    )  # generate ground truth trajectory x (based on ground truth velocity control u)

    z_odom = u + np.random.randn(3, trajectory_length - 1) * noise_odom
    z_icp = u + np.random.randn(3, trajectory_length - 1) * noise_icp
    z_ma, z_mr = np.zeros((3, x.shape[1] - 1)), np.zeros((3, x.shape[1] - 1))
    for t in range(1, x.shape[1]):
        z_ma[:, t - 1] = w2r(ma[:, 0], x[:, t]) + np.random.randn(3) * noise_ma
        z_mr[:, t - 1] = w2r(mr[:, 0], x[:, t]) + np.random.randn(3) * noise_mr

    return x0, x, mr, ma, z_odom, z_ma, z_mr, z_icp


def factor_graph_demo():
    """Generate a demonstration trajectory and run the factorgraph localization on it."""
    required_mr_accuracy = 0.2  # The error of mr localization which is considered OKAY. Change it when changing noise.
    # Generate a test trajectory
    x0, X, Mr, Ma, Z_odom, Z_ma, Z_mr, Z_icp = sim(
        trajectory_length=30,
        v=0.3,
        w=0.3,
        noise_odom=0.1,
        noise_mr=0.1,
        noise_ma=0.1,
        noise_icp=0.02,
    )

    # Set costs for the measurements
    C_odom = 100000 * np.repeat(np.eye(3)[None, :, :], Z_odom.shape[1], axis=0)
    C_mr = 0.0 * np.repeat(np.eye(3)[None, :, :], Z_mr.shape[1], axis=0)
    C_ma = 0.0 * np.repeat(np.eye(3)[None, :, :], Z_ma.shape[1], axis=0)
    C_icp = 0.0 * np.repeat(np.eye(3)[None, :, :], Z_icp.shape[1], axis=0)

    fuse_icp = False
    fg = FactorGraph(
        ma=Ma.ravel().tolist(),
        mr_gt=Mr.ravel().tolist(),
        fuse_icp=fuse_icp,
    )
    state = fg.state

    opt_step = 3  # After how many measurements we perform optimization
    mr_keep_every_nth = 4
    ma_keep_every_nth = 2

    for i in range(X.shape[1] - 1):
        gt = X[:, i].tolist()
        z_odom = Z_odom[:, i]
        z_icp = Z_icp[:, i]
        z_mr = (
            Z_mr[:, i]
            if i % mr_keep_every_nth == 0
            else np.array([np.nan, np.nan, np.nan])
        )
        z_ma = (
            Z_ma[:, i]
            if i % ma_keep_every_nth == 0
            else np.array([np.nan, np.nan, np.nan])
        )
        t = int((i + 1) * 1e9)

        odom_samples = {t: (z_odom, C_odom[i, :, :])}
        icp_samples = {t: (z_icp, C_icp[i, :, :])}

        # Add the measurement
        state = FactorGraph.add_z(
            state,
            fg.config,
            odom_samples,
            z_mr,
            C_mr[i, :, :],
            z_ma,
            C_ma[i, :, :],
            icp_samples,
            gt,
            t,
        )

        # Reoptimize and visualize only after a few measurements
        if i % opt_step != (opt_step - 1):
            continue

        # Run optimization
        state = FactorGraph.optimize(state, fg.config)

        err = np.linalg.norm(state.mr - Mr)
        print("Relative marker localization error: %.2f m." % (err,))
        if err < required_mr_accuracy:
            print("Marker localization is OKAY.")
        else:
            print("Marker localization is NOT OKAY.")

        # Visualize the factorgraph
        FactorGraph.visu(copy.deepcopy(state), fg.config, only_main=True)

        plt.tight_layout()
        plt.pause(1)
        image_name = "traj_%02i.png" % i
        image_path = os.path.join("/tmp", image_name)
        plt.savefig(image_path)
    plt.pause(10)


if __name__ == "__main__":
    # main()
    factor_graph_demo()
