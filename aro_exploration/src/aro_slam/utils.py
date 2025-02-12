from typing import Callable, Any, List, Union, Sequence, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Do not remove, adds support for 3D plots
from numpy.lib.recfunctions import structured_to_unstructured
from timeit import default_timer as timer

try:
    import rospy
    logdebug = rospy.logdebug
except ImportError:
    logdebug = print


def timing(f: Callable):
    """A decorator that can be used to measure and print function run time."""

    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        try:
            ret = f(*args, **kwargs)
            return ret
        finally:
            t1 = timer()
            logdebug('%s %.3f ms' % (f.__name__, (t1 - t0) * 1000))
    return timing_wrapper


def slots(msg: Any) -> List[Any]:
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg: Any) -> np.ndarray:
    """Return message attributes (slots) as Numpy array."""
    return np.array(slots(msg))


def col(arr: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert array to column vector.
    :param arr: The input array or list.
    :return: The Numpy column vector.
    """
    arr = np.asarray(arr)
    return arr.reshape((arr.size, 1))


def logistic(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Standard logistic function, inverse of logit function.
    :param x: The input value(s).
    :return: The logistic function value(s).
    """
    return 1. / (1. + np.exp(-x))


def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Logit function or the log-odds, inverse of logistic function.
    The logarithm of the odds p / (1 - p) where p is probability.
    :param p: The probability(s).
    :return: The log-odds.
    """
    return np.log(p / (1. - p))


def as_unstructured(x: np.ndarray) -> np.ndarray:
    """Convert the input pointcloud array into the unstructured representation if it's not already unstructured.

    :param x: The input pointcloud array (can be structured or unstructured).
    :return: The corresponding unstructured representation.
    """
    x = np.asarray(x)
    if x.dtype.names:
        return structured_to_unstructured(x[['x', 'y', 'z']])
    return x


def visualize_clouds_2d(P: np.ndarray, Q: np.ndarray, title: Optional[str] = None, **kwargs) -> None:
    """Visualize two 2D clouds.
    :param P: The source cloud. It can be structured with xyz fields or unstructured Numpy array Nx2.
    :param Q: The target cloud. It can be structured with xyz fields or unstructured Numpy array Nx2.
    :param title: Title of the plot.
    :param kwargs: Additional kwargs passed to :fun:`matplotlib.pyplot.plot`.
    """
    P = as_unstructured(P)
    Q = as_unstructured(Q)

    plt.figure()

    plt.plot(P[:, 0], P[:, 1], 'o', label='source cloud', **kwargs)
    plt.plot(Q[:, 0], Q[:, 1], 'x', label='target cloud', **kwargs)

    if title is not None:
        plt.title(title)

    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()


def filter_grid(cloud: np.ndarray, grid_res: float, log=False, rng=np.random.default_rng(135)) -> np.ndarray:
    """Filter pointcloud by a uniform grid and keep one point within each occupied cell. Order is not preserved.

    :param cloud: The input pointcloud array. It can be structured with xyz fields or unstructured Numpy array Nx2 or
                  Numpy array Nx3.
    :param grid_res: Resolution of the grid.
    :param log: If True, print debug information about the grid.
    :param rng: Random number generator to use for shuffling the input cloud.
    :return: Subset of `cloud` that contains only the filtered pointcloud.
    """
    assert isinstance(cloud, np.ndarray)
    assert isinstance(grid_res, float) and grid_res > 0.0

    # Convert to numpy array with positions.
    x = as_unstructured(cloud)

    # Create voxel indices.
    keys: List[List[int]] = np.floor(x / grid_res).astype(int).tolist()

    # Last key will be kept, shuffle if needed.
    # Create index array for tracking the input points.
    ind = list(range(len(keys)))

    # Make the last item random.
    rng.shuffle(ind)
    # keys = keys[ind]
    keys = [keys[i] for i in ind]

    # Convert to immutable keys (tuples).
    keys: List[Tuple[int]] = [tuple(i) for i in keys]

    # Dict keeps the last value for each key (already reshuffled).
    key_to_ind = dict(zip(keys, ind))
    ind = list(key_to_ind.values())

    if log:
        logdebug('%.3f = %i / %i points kept (grid res. %.3f m).' % (
            len(ind) / len(keys), len(ind), len(keys), grid_res))

    filtered = cloud[ind]
    return filtered


def visualize_clouds_3d(P: np.ndarray, Q: np.ndarray, title: Optional[str] = None, **kwargs) -> None:
    """Visualize two 3D clouds.
    :param P: The source cloud. It can be structured with xyz fields or unstructured Numpy array Nx3.
    :param Q: The target cloud. It can be structured with xyz fields or unstructured Numpy array Nx3.
    :param title: Title of the plot.
    :param kwargs: Additional kwargs passed to :fun:`matplotlib.pyplot.plot`.
    """
    P = as_unstructured(P)
    Q = as_unstructured(Q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(P[:, 0], P[:, 1], P[:, 2], 'o', label='source cloud', **kwargs)
    ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], 'x', label='target cloud', **kwargs)

    if title is not None:
        ax.set_title(title)

    set_axes_equal(ax)
    ax.grid()
    ax.legend()
    plt.show()


# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax: Axes3D):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc...
    This is one possible solution to Matplotlib's `ax.set_aspect('equal')` and `ax.axis('equal')` not working for 3D.

    :param ax: a matplotlib axis, e.g., as output from `plt.gca()`.
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
