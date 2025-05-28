"""
Generalized implementation of scipy.optimize.least_squares that allows using different sparse Least-squares solvers
than the 3 officially supported (**lm**, **trf**, **dogbox**) to support solving nonlinear least-squares problems.

**trf (Trust-Region)** solver is usually quite fast and good with sparse matrices, but we have observed that sometimes
(mostly when returning to a marker after a long time), the optimization requires quite a lot of iterations, leading
to optimization times over a second.

The **cholmod** solver is a little slower than trf in the general case, but it doesn't suffer from the high variability
in the number of iterations it needs.

Other solvers are implemented too, mostly for teaching purposes:

  - **pinv** computes the LSQ problem using pseudo-inverse, which is the standard method shown in lectures which do not
    care about efficiency. You can see yourself that pinv gets quite quickly completely unusable.
  - The **umfpack** solver was more a try when we were looking for better solvers. It doesn't make much sense to use it
    as it is ~2x slower than cholmod.
"""

import sys
import time
import unittest
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
from numpy.linalg import norm

sys.path.insert(0, "/F/miniforge3/lib/python3.10/site-packages")
from scipy._lib.six import string_types
from scipy.optimize import OptimizeResult
from scipy.optimize._lsq.common import (
    check_termination,
    compute_grad,
    compute_jac_scale,
    in_bounds,
    make_strictly_feasible,
    print_header_nonlinear,
    print_iteration_nonlinear,
    scale_for_robust_loss_function,
)
from scipy.optimize._lsq.dogbox import dogbox
from scipy.optimize._lsq.least_squares import (
    IMPLEMENTED_LOSSES,
    TERMINATION_MESSAGES,
    approx_derivative,
    call_minpack,
    check_jac_sparsity,
    check_tolerance,
    check_x_scale,
    construct_loss_function,
    prepare_bounds,
)

# Optimizers
from scipy.optimize._lsq.trf import trf
from scipy.sparse import csr_matrix, dia_matrix, issparse, spmatrix
from scipy.sparse.linalg import LinearOperator
from sksparse.cholmod import Factor, cholesky, cholesky_AAt

from aro_localization.utils import merge_dicts

JacTypes = Union[np.ndarray, spmatrix, LinearOperator]
"""All types supported as Jacobian."""


class NonlinearLeastSquaresSolver:
    """Common interface for solving nonlinear least-squares problems."""

    def __init__(self, name: str):
        """
        :param name: Name of the solver (used e.g. in parameters selecting the solver).
        """
        self.name: str = name
        """Name of the solver."""

        self.options: Dict[str, Any] = {}
        """Options of the solver."""

    def validate_loss(
        self, loss: Union[str, Callable[[np.ndarray], np.ndarray], LinearOperator]
    ) -> None:
        """
        Validate the `loss_function` argument to :meth:`solve`.

        :raises ValueError: If the given loss type is not supported by this solver.
        """

    def validate_bounds(
        self, lb: Union[float, np.ndarray], ub: Union[float, np.ndarray]
    ) -> None:
        """
        Validate the bounds arguments to :meth:`solve`.

        :raises ValueError: If this solver does not support bounded optimization.
        """

    def validate_dimensions(self, num_residuals: int, num_variables: int) -> None:
        """
        Validate the dimensions of the problem are supported by this solver.

        :param num_residuals: Number of residuals (rows) in the problem.
        :param num_variables: Number of variables (columns) in the problem.
        :raises ValueError: If this solver does not support problems with the given dimensions.
        """

    def validate_jacobian(self, jacobian: Union[np.ndarray, spmatrix]) -> None:
        """
        Validate the Jacobian matrix passed to :meth:`solve` (e.g. whether it is sparse or not).

        :param jacobian: The Jacobian provided by user.
        :raises ValueError: If this solver does not support the given type of Jacobian.
        """

    def adjust_initial_guess(
        self, x0: np.ndarray, lb: Union[float, np.ndarray], ub: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Adjust the initial guess of the optimization problem.

        :param x0: The provided initial guess.
        :param lb: Lower bound(s) of the optimization problem.
        :param ub: Upper bound(s) of the optimization problem.
        :return: The adjusted initial guess of the optimization problem (e.g. `x0` moved inside the bounds).
        """
        return x0

    def set_options(self, options: Dict[str, Any]) -> None:
        """
        Set options of the solver.

        :param options: The new options. They are merged with the current options.
        """
        self.options = merge_dicts(self.options, options)

    def estimate_jacobian(
        self,
        fun: Callable[..., np.ndarray],
        jac: Optional[Union[str, Callable[..., JacTypes]]],
        x0: np.ndarray,
        f0: np.ndarray,
        num_variables: int,
        num_residuals: int,
        jac_sparsity: Optional[Union[np.ndarray, spmatrix]],
        bounds: Tuple[float, float],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[
        Optional[JacTypes],
        Optional[Callable[[np.ndarray, Optional[np.ndarray]], JacTypes]],
    ]:
        """
        Estimate Jacobian using a finite difference method.

        :param fun: The residuals function.
        :param jac: The method for estimating the Jacobian (either '2-point', '3-point' or 'cs').
        :param x0: Initial guess of the optimization problem.
        :param f0: Residuals of the initial guess.
        :param num_variables: Number of variables (Jacobian columns).
        :param num_residuals: Number of residuals (Jacobian rows).
        :param jac_sparsity: Sparsity structure of the Jacobian.
        :param bounds: Bounds of the problem.
        :param args: Args passed to `fun` and `jac`.
        :param kwargs: Kwargs passed to `fun` and `jac`.
        :return: A tuple containing the Jacobian of the initial guess and a function taking in variable values and
                 possibly the evaluated residuals and returning an estimate of the Jacobian.
        """
        raise NotImplementedError

    def solve(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray, Optional[np.ndarray]], JacTypes],
        x0: np.ndarray,
        f0: np.ndarray,
        J0: JacTypes,
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        ftol: Optional[float],
        xtol: Optional[float],
        gtol: Optional[float],
        max_nfev: Optional[int],
        x_scale: Union[str, float],
        loss_function: Union[str, Callable[[np.ndarray], np.ndarray]],
        verbose: Optional[int],
        initial_fev_time_ms: float,
        initial_jev_time_ms: float,
    ) -> OptimizeResult:
        """
        Solve the nonlinear least-squares problem.

        :param fun: The function returning residuals (its input are current variable estimates). The returned residuals
                    are Numpy array N (one number for each residual).
        :param jac: The function returning Jacobians of the residuals w.r.t. all variables (its input are current
                    variable estimates and possibly evaluated residuals). The returned Jacobians are
                    Numpy array NxM (rows are residuals, columns are variables).
        :param x0: Initial guess of the variable values. Numpy array M.
        :param f0: Residual values for `x0`. Numpy array N.
        :param J0: Jacobian for `x0`. Numpy array NxM (or a sparse matrix).
        :param lb: Lower bound(s). Pass `-np.inf` for unbounded problems. Either one number, or one per variable.
        :param ub: Upper bound(s). Pass `np.inf` for unbounded problems. Either one number, or one per variable.
        :param ftol: Convergence limit for changes in residual values.
        :param xtol: Convergence limit for changes in variable values.
        :param gtol: Convergence limit for changes in gradient values.
        :param max_nfev: Maximum number of iterations.
        :param x_scale: Scaling factor for variable values (either float or 'jac' to scale by Jacobian values).
        :param loss_function: Loss function to be minimized (either 'linear', 'soft_l1', 'huber', 'cauchy' or callback).
        :param verbose: {0, 1, 2}, the level of output verbosity. 0 means silent.
        :param initial_fev_time_ms: How long did it take to compute f0 (used in verbosity 2 mode).
        :param initial_jev_time_ms: How long did it take to compute J0 (used in verbosity 2 mode).
        :return: :class:`OptimizeResult` with the same fields as the result of
                 :meth:`scipy.optimize.least_squares.least_squares`.
        """
        raise NotImplementedError

    def __str__(self):
        return "%s %r" % (self.name, self.options)


class LMSolver(NonlinearLeastSquaresSolver):
    """
    Original 'lm' solver from scipy.

    Accepted options:

      - diff_step (float, default is automatic): Size of the automatic differentiation step.
    """

    def __init__(self):
        super().__init__("lm")
        self.diff_step = None

    def set_options(self, options: Dict[str, Any]) -> None:
        super().set_options(options)
        self.diff_step: Union[None, np.ndarray, Iterable[float], int, float] = (
            options.get("diff_step")
        )

    def validate_loss(
        self, loss: Union[str, Callable[[np.ndarray], np.ndarray], LinearOperator]
    ) -> None:
        if loss != "linear":
            raise ValueError("method='lm' supports only 'linear' loss function.")

    def validate_bounds(
        self, lb: Union[float, np.ndarray], ub: Union[float, np.ndarray]
    ) -> None:
        if not np.all((lb == -np.inf) & (ub == np.inf)):
            raise ValueError("Method 'lm' doesn't support bounds.")

    def validate_dimensions(self, num_residuals: int, num_variables: int) -> None:
        if num_residuals < num_variables:
            raise ValueError(
                "Method 'lm' doesn't work when the number of residuals is less than the number of variables."
            )

    def validate_jacobian(self, jacobian: JacTypes) -> None:
        if not isinstance(jacobian, np.ndarray):
            raise ValueError("method='lm' works only with dense Jacobian matrices.")

    def estimate_jacobian(
        self,
        fun: Callable[..., np.ndarray],
        jac: Optional[Union[str, Callable[..., JacTypes]]],
        x0: np.ndarray,
        f0: np.ndarray,
        num_variables: int,
        num_residuals: int,
        jac_sparsity: Optional[Union[np.ndarray, spmatrix]],
        bounds: Tuple[float, float],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[
        Optional[JacTypes],
        Optional[Callable[[np.ndarray, Optional[np.ndarray]], JacTypes]],
    ]:
        if jac_sparsity is not None:
            raise ValueError("method='lm' does not support `jac_sparsity`.")

        if jac != "2-point":
            warn(f"jac='{jac}' works equivalently to '2-point' for method='lm'.")

        return None, None

    def solve(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray, Optional[np.ndarray]], JacTypes],
        x0: np.ndarray,
        f0: np.ndarray,
        J0: JacTypes,
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        x_scale: Union[str, float],
        loss_function: Union[str, Callable[[np.ndarray], np.ndarray]],
        verbose: Optional[int],
        initial_fev_time_ms: float,
        initial_jev_time_ms: float,
    ) -> OptimizeResult:
        return call_minpack(
            fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, self.diff_step
        )


class TRFSolverBase(NonlinearLeastSquaresSolver):
    """
    Base class for Trust-Region-based solvers.

    Accepted options:

      - solver (str, default is exact for dense and lsmr for sparse): Which Trust-Region solver to use (exact, lsmr).
      - diff_step (float, default is automatic): Size of the automatic differentiation step.
      - all other options are passed to the trf() or dogbox() as `tr_options`.
    """

    def __init__(self, name):
        super().__init__(name)
        self.solver: Optional[str] = None
        self.diff_step: Union[None, np.ndarray, Iterable[float], int, float] = None
        self.tr_options: Dict[str, Any] = {}

    def validate_jacobian(self, jacobian: JacTypes) -> None:
        if not isinstance(jacobian, np.ndarray) and self.solver == "exact":
            raise ValueError(
                "tr_solver='exact' works only with dense Jacobian matrices."
            )

        if self.solver is None:
            if isinstance(jacobian, np.ndarray):
                self.solver = "exact"
            else:
                self.solver = "lsmr"

    def set_options(self, options: Dict[str, Any]) -> None:
        super().set_options(options)
        self.diff_step = options.pop("diff_step", None)
        solver = options.pop("tr_solver", None)
        if solver not in [None, "exact", "lsmr"]:
            raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")
        self.solver = solver
        self.tr_options = options

    def estimate_jacobian(
        self,
        fun: Callable[..., np.ndarray],
        jac: Optional[Union[str, Callable[..., JacTypes]]],
        x0: np.ndarray,
        f0: np.ndarray,
        num_variables: int,
        num_residuals: int,
        jac_sparsity: Optional[Union[np.ndarray, spmatrix]],
        bounds: Tuple[float, float],
        args: Sequence[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[
        Optional[JacTypes],
        Optional[Callable[[np.ndarray, Optional[np.ndarray]], JacTypes]],
    ]:
        if jac_sparsity is not None and self.solver == "exact":
            raise ValueError("tr_solver='exact' is incompatible with `jac_sparsity`.")

        jac_sparsity = check_jac_sparsity(jac_sparsity, num_variables, num_residuals)

        def jac_wrapped(x, f):
            J = approx_derivative(
                fun,
                x,
                rel_step=self.diff_step,
                method=jac,
                f0=f,
                bounds=bounds,
                args=args,
                kwargs=kwargs,
                sparsity=jac_sparsity,
            )
            if J.ndim != 2:  # J is guaranteed not sparse.
                J = np.atleast_2d(J)

            return J

        return jac_wrapped(x0, f0), jac_wrapped


class TRFSolver(TRFSolverBase):
    """Original 'trf' solver from scipy."""

    def __init__(self):
        super().__init__("trf")

    def adjust_initial_guess(
        self, x0: np.ndarray, lb: Union[float, np.ndarray], ub: Union[float, np.ndarray]
    ) -> np.ndarray:
        return make_strictly_feasible(x0, lb, ub)

    def solve(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray, Optional[np.ndarray]], JacTypes],
        x0: np.ndarray,
        f0: np.ndarray,
        J0: JacTypes,
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        x_scale: Union[str, float],
        loss_function: Union[str, Callable[[np.ndarray], np.ndarray]],
        verbose: Optional[int],
        initial_fev_time_ms: float,
        initial_jev_time_ms: float,
    ) -> OptimizeResult:
        return trf(
            fun,
            jac,
            x0,
            f0,
            J0,
            lb,
            ub,
            ftol,
            xtol,
            gtol,
            max_nfev,
            x_scale,
            loss_function,
            self.solver,
            self.tr_options.copy(),
            verbose,
        )


class DogboxSolver(TRFSolverBase):
    """Original 'dogbox' solver from scipy."""

    def __init__(self):
        super().__init__("dogbox")

    def solve(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray, Optional[np.ndarray]], JacTypes],
        x0: np.ndarray,
        f0: np.ndarray,
        J0: JacTypes,
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        x_scale: Union[str, float],
        loss_function: Union[str, Callable[[np.ndarray], np.ndarray]],
        verbose: Optional[int],
        initial_fev_time_ms: float,
        initial_jev_time_ms: float,
    ) -> OptimizeResult:
        if self.solver == "lsmr" and "regularize" in self.options:
            warn(
                "The keyword 'regularize' in `tr_options` is not relevant for 'dogbox' method."
            )
            tr_options = self.options.copy()
            del tr_options["regularize"]
            self.options = tr_options

        return dogbox(
            fun,
            jac,
            x0,
            f0,
            J0,
            lb,
            ub,
            ftol,
            xtol,
            gtol,
            max_nfev,
            x_scale,
            loss_function,
            self.solver,
            self.tr_options,
            verbose,
        )


class CustomSolverBase(NonlinearLeastSquaresSolver):
    """Base class for custom least-squares solvers."""

    def __init__(self, name: str):
        """
        :param name: Name of the solver (used e.g. in parameters selecting the solver).
        """
        super().__init__(name)

    def validate_bounds(
        self, lb: Union[float, np.ndarray], ub: Union[float, np.ndarray]
    ) -> None:
        if not np.all((lb == -np.inf) & (ub == np.inf)):
            raise ValueError("Custom solvers don't support bounds.")

    def solve_dx(self, J: JacTypes, fx: np.ndarray) -> np.ndarray:
        """
        Solve `x` from :math:`J^T J x = J^T \\mathrm{fx}`. Child classes should override this method.

        :param J: Jacobian.
        :param fx: Residuals.
        :return: Variable values satisfying the above equation.
        """
        raise NotImplementedError

    # This function is heavily based on the implementation of scipy.optimize._lsq.trf .
    # It behaves the same for the original methods, but contains a few performance optimizations for working with
    # sparse matrices and also offers more verbose debug printing regarding performance.
    def solve(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        jac: Callable[[np.ndarray, Optional[np.ndarray]], JacTypes],
        x0: np.ndarray,
        f0: np.ndarray,
        J0: JacTypes,
        lb: Union[float, np.ndarray],
        ub: Union[float, np.ndarray],
        ftol: float,
        xtol: float,
        gtol: float,
        max_nfev: int,
        x_scale: Union[str, float],
        loss_function: Union[str, Callable[[np.ndarray], np.ndarray]],
        verbose: Optional[int],
        initial_fev_time_ms: float,
        initial_jev_time_ms: float,
    ) -> OptimizeResult:
        x = x0.copy()

        f = f0
        f_true = f.copy()
        nfev = 1

        J = J0
        njev = 1

        if loss_function is not None:
            rho = loss_function(f)
            cost = 0.5 * np.sum(rho[0])
            J, f = scale_for_robust_loss_function(J, f, rho)
        else:
            cost = 0.5 * np.dot(f, f)

        g = compute_grad(J, f)

        jac_scale = isinstance(x_scale, string_types) and x_scale == "jac"
        if jac_scale:
            scale, scale_inv = compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale

        if max_nfev is None:
            max_nfev = x0.size * 100

        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None

        if verbose == 2:
            print_header_nonlinear()

        tic = time.perf_counter()
        times = []
        lsq_times = []
        fev_times = [initial_fev_time_ms]
        jev_times = [initial_jev_time_ms]
        while True:
            g_norm = norm(g, ord=np.inf)
            if g_norm < gtol:
                termination_status = 1

            if verbose == 2:
                print_iteration_nonlinear(
                    iteration, nfev, cost, actual_reduction, step_norm, g_norm
                )

            if termination_status is not None or nfev == max_nfev:
                times.append((time.perf_counter() - tic) * 1e3)
                break

            tic = time.perf_counter()

            J_h = J @ dia_matrix((scale, (0,)), shape=(len(scale), len(scale)))

            tic_lsq = time.perf_counter()
            step_h = self.solve_dx(J_h, f)
            lsq_times.append((time.perf_counter() - tic_lsq) * 1e3)

            step = scale * step_h
            x_new = x + step
            tic_fev = time.perf_counter()
            f_new = fun(x_new)
            fev_times.append((time.perf_counter() - tic_fev) * 1e3)
            nfev += 1

            # Usual trust-region step quality estimation.
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), 1.0, ftol, xtol
            )

            if actual_reduction > 0:
                x = x_new

                f = f_new
                f_true = f.copy()

                cost = cost_new

                tic_jev = time.perf_counter()
                J = jac(x, f)
                jev_times.append((time.perf_counter() - tic_jev) * 1e3)
                njev += 1

                if loss_function is not None:
                    rho = loss_function(f)
                    J, f = scale_for_robust_loss_function(J, f, rho)

                g = compute_grad(J, f)

                if jac_scale:
                    scale, scale_inv = compute_jac_scale(J, scale_inv)
            else:
                step_norm = 0
                actual_reduction = 0

            iteration += 1
            times.append((time.perf_counter() - tic) * 1e3)

        if verbose == 2:
            print(
                "Iteration times in ms: " + ", ".join(["%0.2f" % (t,) for t in times])
            )
            print(
                "Least Sq. times in ms: "
                + ", ".join(["%0.2f" % (t,) for t in lsq_times])
            )
            print(
                "Res. Eval times in ms: "
                + ", ".join(["%0.2f" % (t,) for t in fev_times])
            )
            print(
                "Jac. Eval times in ms: "
                + ", ".join(["%0.2f" % (t,) for t in jev_times])
            )

        if termination_status is None:
            termination_status = 0

        active_mask = np.zeros_like(x)
        return OptimizeResult(
            x=x,
            cost=cost,
            fun=f_true,
            jac=J,
            grad=g,
            optimality=g_norm,
            active_mask=active_mask,
            nfev=nfev,
            njev=njev,
            status=termination_status,
        )


class CholmodSolver(CustomSolverBase):
    """
    Least squares solver using the cholmod library.

    Accepted options:

      - *beta* (float, default 1e-10): Scale of the unit matrix added to Jt*J diagonal.
      - *use_AAT_decomposition* (bool, default True): Decompose Jt*J directly instead of computing the product.
      - *ordering_method* (str, default "default"): Row-ordering method. See :func:`sksparse.cholmod.cholesky`.
      - *solve_method* (str, default "a"): Which cholmod method should be used for solving the least squares equation.

        - a, ldlt, l_dlt, l_lt
        - a should be the fastest
    """

    def __init__(self):
        super().__init__("cholmod")
        self.beta: float = 1e-10
        """This is scale of the unit matrix added to Jt*J diagonal to help with non-full-rank matrix decomposition."""

        self.use_AAt_decomposition: bool = True
        """True should be faster (it decomposes the Jt*J product without explicitly computing it)."""

        self.ordering_method: str = "default"
        """Row ordering for the Cholesky decomposition. default, natural, best, amd, metis, nesdis, colamd."""

        self.solve_method = "a"
        """Which cholmod method should be used for solving the least squares equation. a, ldlt, l_dlt, l_lt. a should
        be the fastest."""

    def set_options(self, options: Dict[str, Any]) -> None:
        allowed_ordering_methods = (
            "default",
            "natural",
            "best",
            "amd",
            "metis",
            "nesdis",
            "colamd",
        )
        if (
            options.get("ordering_method", self.ordering_method)
            not in allowed_ordering_methods
        ):
            raise ValueError(
                "Ordering method {} is not allowed. It must be one of {}.".format(
                    options["ordering_method"], ",".join(allowed_ordering_methods)
                )
            )
        allowed_solve_methods = ("a", "ldlt", "l_dlt", "l_lt")
        if options.get("solve_method", self.solve_method) not in allowed_solve_methods:
            raise ValueError(
                "Solve method {} is not allowed. It must be one of {}.".format(
                    options["solve_method"], ",".join(allowed_solve_methods)
                )
            )

        super().set_options(options)
        self.beta = float(options.pop("beta", self.beta))
        self.use_AAt_decomposition = bool(
            options.pop("use_AAt_decomposition", self.use_AAt_decomposition)
        )
        self.ordering_method = str(options.pop("ordering_method", self.ordering_method))
        self.solve_method = str(options.pop("solve_method", self.solve_method))

    def decompose_JTJ(self, JT: JacTypes) -> Factor:
        """
        Compute the Cholesky decomposition of the JT @ JT.T product.

        :param JT: Transposed Jacobian.
        :return: The decomposition.
        """
        if self.use_AAt_decomposition:
            return cholesky_AAt(
                JT, beta=self.beta, ordering_method=self.ordering_method
            )
        return cholesky(JT @ JT.T, beta=self.beta, ordering_method=self.ordering_method)

    def solve_dx(self, J: JacTypes, fx: np.ndarray) -> np.ndarray:
        JT = J.T
        factor = self.decompose_JTJ(JT)

        if self.solve_method == "a":
            dx = factor.solve_A(JT @ fx)
        elif self.solve_method == "ldlt":
            if self.ordering_method == "natural":
                dx = factor.solve_LDLt(JT @ fx)
            else:
                p = factor.P()
                dx = np.zeros((JT.shape[0],))
                dx[p] = factor.solve_LDLt((JT @ fx)[p])
        elif self.solve_method == "l_dlt":
            if self.ordering_method == "natural":
                y = factor.solve_L(JT @ fx, use_LDLt_decomposition=True)
                dx = factor.solve_DLt(y)
            else:
                p = factor.P()
                dx = np.zeros((JT.shape[0],))
                y = factor.solve_L((JT @ fx)[p], use_LDLt_decomposition=True)
                dx[p] = factor.solve_DLt(y)
        elif self.solve_method == "l_lt":
            if self.ordering_method == "natural":
                y = factor.solve_L(JT @ fx, use_LDLt_decomposition=False)
                dx = factor.solve_Lt(y, use_LDLt_decomposition=False)
            else:
                p = factor.P()
                dx = np.zeros((JT.shape[0],))
                y = factor.solve_L((JT @ fx)[p], use_LDLt_decomposition=False)
                dx[p] = factor.solve_Lt(y, use_LDLt_decomposition=False)
        else:
            raise ValueError(f"Solve method {self.solve_method} is not supported.")

        return -dx


class PinvSolver(CustomSolverBase):
    """Least squares solver using pseudo-inverse."""

    def __init__(self):
        super().__init__("pinv")

    def solve_dx(self, J: JacTypes, fx: np.ndarray) -> np.ndarray:
        return -np.linalg.pinv(np.array(J.todense())) @ fx


SOLVERS = {
    "lm": LMSolver(),
    "trf": TRFSolver(),
    "dogbox": DogboxSolver(),
    "cholmod": CholmodSolver(),
    "pinv": PinvSolver(),
}


try:
    # pip3 install scikit-umfpack
    import scikits
    import scikits.umfpack

    has_umfpack = True
except ImportError:
    has_umfpack = False

if has_umfpack:

    class UmfpackSolver(CustomSolverBase):
        """
        Least squares solver using umfpack library.

        Accepted options:

          - *beta* (float, default 1e-10): Scale of the unit matrix added to Jt*J diagonal.
        """

        def __init__(self):
            super().__init__("umfpack")
            self.beta: float = 1e-10
            """This is scale of the unit matrix added to Jt*J diagonal to help with non-full-rank matrix
            decomposition."""

        def set_options(self, options: Dict[str, Any]) -> None:
            super().set_options(options)
            self.beta = float(options.pop("beta", self.beta))

        def solve_dx(self, J: JacTypes, fx: np.ndarray) -> np.ndarray:
            JT = J.T
            I = dia_matrix(
                (np.ones((JT.shape[0],)), (0,)), shape=(JT.shape[0], JT.shape[0])
            )
            return -scikits.umfpack.spsolve(JT @ JT.T + self.beta * I, JT @ fx)

    SOLVERS["umfpack"] = UmfpackSolver()


# This function has been copied from scipy.optimize._lsq.least_squares and adapted to support additional solvers .
def least_squares(
    fun: Callable[..., np.ndarray],
    x0: np.ndarray,
    jac: Optional[Union[str, Callable[..., JacTypes]]] = "2-point",
    bounds: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]] = (
        -np.inf,
        np.inf,
    ),
    method: Union[str, NonlinearLeastSquaresSolver] = "trf",
    ftol: Optional[float] = 1e-8,
    xtol: Optional[float] = 1e-8,
    gtol: Optional[float] = 1e-8,
    x_scale: Union[str, float] = 1.0,
    loss: Union[str, Callable[[np.ndarray], np.ndarray], LinearOperator] = "linear",
    f_scale: float = 1.0,
    diff_step: Union[None, np.ndarray, Iterable[float], float] = None,
    tr_solver: Optional[str] = None,
    tr_options={},
    jac_sparsity: Optional[Union[np.ndarray, spmatrix]] = None,
    max_nfev: Optional[int] = None,
    verbose: Optional[int] = 0,
    args=(),
    kwargs={},
):
    """
    Solve a nonlinear least-squares problem with bounds on the variables.

    Given the residuals `f(x)` (an m-dimensional real function of n real variables) and the loss function `rho(s)`
    (a scalar function), `least_squares` finds a local minimum of the cost function `F(x)`::

        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
        subject to lb <= x <= ub

    The purpose of the loss function `rho(s)` is to reduce the influence of outliers on the solution.

    :param fun: Function which computes the vector of residuals, with the signature ``fun(x, *args, **kwargs)``, i.e.,
                the minimization proceeds with respect to its first argument. The argument ``x`` passed to this
                function is an ndarray of shape (n,) (never a scalar, even for n=1). It must return a 1-d array_like
                of shape (m,) or a scalar. If the argument ``x`` is complex or the function ``fun`` returns complex
                residuals, it must be wrapped in a real function of real arguments.
    :param x0: array_like with shape (n,) or float.  Initial guess on independent variables. If float, it will be
               treated as a 1-d array with one element.
    :param jac: {'2-point', '3-point', 'cs', callable}, optional. Method of computing the Jacobian matrix (an m-by-n
                matrix, where element (i, j) is the partial derivative of `f[i]` with respect to `x[j]`).
                The keywords select a finite difference scheme for numerical estimation. The scheme '3-point' is more
                accurate, but requires twice as many operations as '2-point' (default). The scheme 'cs' uses complex
                steps, and while potentially the most accurate, it is applicable only when `fun` correctly handles
                complex inputs and can be analytically continued to the complex plane. Method 'lm' always uses the
                '2-point' scheme. If callable, it is used as ``jac(x, *args, **kwargs)`` and should return a good
                approximation (or the exact value) for the Jacobian as an array_like (np.atleast_2d is applied),
                a sparse matrix or a `scipy.sparse.linalg.LinearOperator`.
    :param bounds: 2-tuple of array_like, optional. Lower and upper bounds on independent variables. Defaults to no
                   bounds. Each array must match the size of `x0` or be a scalar, in the latter case a bound will be
                   the same for all variables. Use ``np.inf`` with an appropriate sign to disable bounds on all or some
                   variables.
    :param method: {'trf', 'dogbox', 'lm', 'cholmod', 'pinv', 'umfpack'} or NonlinearLeastSquaresSolver instance.
                   Algorithm to perform minimization. Default is 'trf'. See Notes for more information.
    :param ftol: Tolerance for termination by the change of the cost function. Default is 1e-8. The optimization process
                 is stopped when ``dF < ftol * F``, and there was an adequate agreement between a local quadratic model
                 and the true model in the last step. If None, the termination by this condition is disabled.
    :param xtol: Tolerance for termination by the change of the independent variables. Default is 1e-8.
                 For most methods, the exact condition is ``norm(dx) < xtol * (xtol + norm(x))``. If None, the
                 termination by this condition is disabled.
    :param gtol: Tolerance for termination by the norm of the gradient. Default is 1e-8. For most method, the exact
                 condition is ``norm(g_scaled, ord=np.inf) < gtol``, where ``g_scaled`` is the value of the gradient
                 scaled to account for the presence of the bounds. If None, the termination by this condition is
                 disabled.
    :param x_scale: Characteristic scale of each variable. Setting `x_scale` is equivalent to reformulating the problem
                    in scaled variables ``xs = x / x_scale``. An alternative view is that the size of a trust region
                    along j-th dimension is proportional to ``x_scale[j]``. Improved convergence may be achieved by
                    setting `x_scale` such that a step of a given size along any of the scaled variables has a similar
                    effect on the cost function. If set to 'jac', the scale is iteratively updated using the inverse
                    norms of the columns of the Jacobian matrix.
    :param loss: Determines the loss function. The following keyword values are allowed: 'linear' (default), 'soft_l1',
                 'huber', 'cauchy', 'arctan'. If callable, it must take a 1-d ndarray ``z=f**2`` and return an
                 array_like with shape (3, m) where row 0 contains function values, row 1 contains first derivatives and
                 row 2 contains second derivatives. Method 'lm' supports only 'linear' loss.
    :param f_scale: Value of soft margin between inlier and outlier residuals, default is 1.0. The loss function is
                    evaluated as follows ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`, and ``rho``
                    is determined by `loss` parameter. This parameter has no effect with ``loss='linear'``, but for
                    other `loss` values it is of crucial importance.
    :param diff_step: Determines the relative step size for the finite difference approximation of the Jacobian.
                      The actual step is computed as ``x * diff_step``. If None (default), then `diff_step` is taken to
                      be a conventional "optimal" power of machine epsilon for the finite difference scheme used.
    :param tr_solver: {None, 'exact', 'lsmr'}. Method for solving trust-region subproblems, relevant only for 'trf'
                      and 'dogbox' methods. If None (default) the solver is chosen based on the type of Jacobian
                      returned on the first iteration.
    :param tr_options: Options passed to the `method` and to `tr_solver` (in case of `trf` method).
    :param jac_sparsity: Defines the sparsity structure of the Jacobian matrix for finitedifference estimation, its
                         shape must be (m, n). If the Jacobian has only few non-zero elements in *each* row, providing
                         the sparsitystructure will greatly speed up the computations. A zero entry means that a
                         corresponding element in the Jacobian is identically zero. If provided, forces the use of
                         'lsmr' trust-region solver. If None (default) then dense differencing will be used. Has no
                         effect for unsupported methods.
    :param max_nfev: Maximum number of function evaluations before the termination.If None (default), the value is
                     chosen automatically.
    :param verbose: {0, 1, 2}, optional. Level of algorithm's verbosity: 0 (default): work silently, 1: display a
                    termination report, 2: display progress during iterations (not supported by 'lm' method).
    :param args: Additional positional arguments passed to `fun` and `jac`. Empty by default.
    :param kwargs: Additional keyword arguments passed to `fun` and `jac`. Empty by default.
    :return: :class:`OptimizeResult` with the same fields as the result of
             :meth:`scipy.optimize.least_squares.least_squares`.
    """
    if method not in SOLVERS and not isinstance(method, NonlinearLeastSquaresSolver):
        raise ValueError(
            "`method` must be {0} or an instance of NonlinearLeastSquaresSolver.".format(
                ", ".join(SOLVERS.keys())
            )
        )

    if isinstance(method, string_types):
        method = SOLVERS[method]

    if jac not in ["2-point", "3-point", "cs"] and not callable(jac):
        raise ValueError("`jac` must be '2-point', '3-point', 'cs' or callable.")

    options = tr_options.copy()
    if tr_solver is not None:
        options["tr_solver"] = tr_solver
    if diff_step is not None:
        options["diff_step"] = diff_step
    method.set_options(options)

    if loss not in IMPLEMENTED_LOSSES and not callable(loss):
        raise ValueError(
            f"`loss` must be one of {IMPLEMENTED_LOSSES.keys()} or a callable."
        )

    method.validate_loss(loss)

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")

    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    if np.iscomplexobj(x0):
        raise ValueError("`x0` must be real.")

    x0 = np.atleast_1d(x0).astype(float)

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    lb, ub = prepare_bounds(bounds, x0.shape[0])
    method.validate_bounds(lb, ub)

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    if np.any(lb >= ub):
        raise ValueError(
            "Each lower bound must be strictly less than each upper bound."
        )

    if not in_bounds(x0, lb, ub):
        raise ValueError("`x0` is infeasible.")

    x_scale = check_x_scale(x_scale, x0)

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)

    def fun_wrapped(x):
        return np.atleast_1d(fun(x, *args, **kwargs))

    x0 = method.adjust_initial_guess(x0, lb, ub)

    tic = time.perf_counter()
    f0 = fun_wrapped(x0)
    initial_fev_time_ms = (time.perf_counter() - tic) * 1e3

    if f0.ndim != 1:
        raise ValueError("`fun` must return at most 1-d array_like.")

    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")

    n = x0.size
    m = f0.size

    method.validate_dimensions(m, n)

    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        if rho.shape != (3, m):
            raise ValueError("The return value of `loss` callable has wrong shape.")
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)

    tic = time.perf_counter()
    if callable(jac):
        J0 = jac(x0, *args, **kwargs)

        if issparse(J0):
            J0 = csr_matrix(J0)

            def jac_wrapped(x, _=None):
                return csr_matrix(jac(x, *args, **kwargs))

        elif isinstance(J0, LinearOperator):

            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs)

        else:
            J0 = np.atleast_2d(J0)

            def jac_wrapped(x, _=None):
                return np.atleast_2d(jac(x, *args, **kwargs))

    else:  # Estimate Jacobian by finite differences.
        J0, jac_wrapped = method.estimate_jacobian(
            fun, jac, x0, f0, m, n, jac_sparsity, bounds, args, kwargs
        )
    initial_jev_time_ms = (time.perf_counter() - tic) * 1e3

    if J0 is not None:
        if J0.shape != (m, n):
            raise ValueError(
                f"The return value of `jac` has wrong shape: expected {(m, n)}, actual {J0.shape}."
            )

        method.validate_jacobian(J0)

        jac_scale = isinstance(x_scale, string_types) and x_scale == "jac"
        if isinstance(J0, LinearOperator) and jac_scale:
            raise ValueError(
                "x_scale='jac' can't be used when `jac` returns LinearOperator."
            )

    result = method.solve(
        fun_wrapped,
        jac_wrapped,
        x0,
        f0,
        J0,
        lb,
        ub,
        ftol,
        xtol,
        gtol,
        max_nfev,
        x_scale,
        loss_function,
        verbose,
        initial_fev_time_ms,
        initial_jev_time_ms,
    )

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0
    result.method = method

    nnz = np.prod(J0.shape)
    if isinstance(J0, spmatrix):
        nnz = J0.nnz
    elif isinstance(jac_sparsity, spmatrix):
        nnz = jac_sparsity.nnz
    elif J0 is not None:
        nnz = np.count_nonzero(J0)
    elif jac_sparsity is not None:
        nnz = np.count_nonzero(jac_sparsity)
    jac_density = nnz / float(np.prod(J0.shape)) if np.prod(J0.shape) > 0 else np.nan

    if verbose >= 1:
        print(result.message)
        print(
            f"Function evaluations {result.nfev}, initial cost {initial_cost:.4e}, final cost {result.cost:.4e}, first-order optimality {result.optimality:.2e}, "
            f"Jacobian density {jac_density:0.4f}.",
        )

    return result


class TestLeastSquares(unittest.TestCase):
    class LSQ:
        def __init__(self, size=200):
            from aro_localization.utils import forward

            self.dt = 0.2

            self.u_k = np.zeros((size, 3))
            self.u_k[:, 0] += 1

            self.p_k = forward(self.u_k.T * self.dt, (0.0, 0.0, 0.0)).T

            self.x0 = np.array(self.p_k)
            self.x0[:, 0] += 0.05 * np.random.random((self.x0.shape[0],))
            self.x0[:, 1] += 0.01 * np.random.random((self.x0.shape[0],))
            self.x0[:, 2] += 0.3 * np.random.random((self.x0.shape[0],))

        def res(self, x: np.ndarray):
            return x - self.p_k.ravel()

        def jac(self, x: np.ndarray):
            J = self.dense_jac(x)
            return csr_matrix(J)

        def dense_jac(self, x: np.ndarray):
            J = np.eye(x.shape[0])
            return J

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fast_test = True
        if fast_test:
            self.default_iters = 1
            self.default_sizes = (20, 200)
        else:
            self.default_iters = 50
            self.default_sizes = (20, 200, 500, 1000)

    def do_tst(self, method, iters, sizes, dense_jac=False, *args, **kwargs):
        text = ""
        for size in sizes:
            lsq = self.LSQ(size)
            jac_sparsity = None
            if dense_jac:
                jac_sparsity = lsq.dense_jac(lsq.x0.ravel()) != 0
            tic = time.perf_counter()
            for i in range(iters):
                sol = least_squares(
                    lsq.res,
                    lsq.x0.ravel(),
                    lsq.dense_jac if dense_jac else lsq.jac,
                    method=method,
                    x_scale="jac",
                    jac_sparsity=jac_sparsity,
                    verbose=0,
                    *args,
                    **kwargs,
                )
            text += f"{size}:{(time.perf_counter() - tic) / iters * 1000:.2f}, "
            self.assertTrue(sol.success)
            self.assertTrue(np.allclose(lsq.p_k.ravel(), sol.x, rtol=1e-4, atol=1e-4))
        print(f"{sol.method}: size:time/it [ms] " + text)

    def test_lm(self):
        self.do_tst(
            "lm",
            dense_jac=True,
            iters=min(4, self.default_iters),
            sizes=[s for s in self.default_sizes if s <= 500],
        )

    def test_trf(self):
        self.do_tst("trf", iters=self.default_iters, sizes=self.default_sizes)

    def test_trf_exact(self):
        self.do_tst(
            "trf",
            iters=min(10, self.default_iters),
            sizes=[s for s in self.default_sizes if s <= 200],
            dense_jac=True,
            tr_solver="exact",
        )

    def test_dogbox(self):
        self.do_tst("dogbox", iters=self.default_iters, sizes=self.default_sizes)

    def test_umfpack(self):
        if not has_umfpack:
            return
        self.do_tst("umfpack", iters=self.default_iters, sizes=self.default_sizes)

    def test_pinv(self):
        self.do_tst(
            "pinv",
            iters=min(2, self.default_iters),
            sizes=[s for s in self.default_sizes if s <= 200],
        )

    def test_cholmod(self):
        self.do_tst("cholmod", iters=self.default_iters, sizes=self.default_sizes)


if __name__ == "__main__":
    unittest.main()
