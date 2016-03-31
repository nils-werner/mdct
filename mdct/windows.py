""" Module for windowing functions not found in SciPy

"""

from __future__ import division
import numpy as np
from scipy.signal import kaiser

__all__ = [
    'kaiser_derived',
]


def kaiser_derived(M, beta):
    """ Return a Kaiser-Bessel derived window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    beta : float
        Kaiser-Bessel window shape parameter.

    Returns
    -------
    w : ndarray
        The window, normalized to fulfil the Princen-Bradley condition.

    Notes
    -----
    This window is only defined for an even number of taps.

    References
    ----------
    .. [1] Wikipedia, "Kaiser window",
           https://en.wikipedia.org/wiki/Kaiser_window

    """
    try:
        from scipy.signal import kaiser_derived as scipy_kd
        return scipy_kd(M, beta)
    except ImportError:
        pass

    if M < 1:
        return np.array([])

    if M % 2:
        raise ValueError(
            "Kaiser Bessel Derived windows are only defined for even number "
            "of taps"
        )

    w = np.zeros(M)
    kaiserw = kaiser(M // 2 + 1, beta)
    csum = np.cumsum(kaiserw)
    halfw = np.sqrt(csum[:-1] / csum[-1])
    w[:M//2] = halfw
    w[-M//2:] = halfw[::-1]

    return w
