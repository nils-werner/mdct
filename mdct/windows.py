""" Module for windowing functions not found in SciPy

"""

from __future__ import division
import numpy as np
from scipy.signal import kaiser

__all__ = [
    'kaiser_derived',
]


def kaiser_derived(M, beta=4.):
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
        The window, with the maximum value normalized to 1.

    Notes
    -----
    This window is only defined for an even number of taps.

    .. versionadded:: 0.16.0

    References
    ----------
    .. [1] Wikipedia, "Kaiser window",
           https://en.wikipedia.org/wiki/Kaiser_window

    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')

    if M % 2:
        raise ValueError(
            "Kaiser Bessel Derived windows are only defined for even number ",
            "of taps"
        )

    w = np.zeros(M)
    tmp = kaiser(M // 2 + 1, beta)[:M // 2]
    num = np.cumsum(tmp)
    denom = np.sum(tmp)

    w[:M//2] = np.sqrt(num / denom)
    w[-M//2:] = np.sqrt(num[::-1] / denom)

    return w
