""" Module for calculating DCT type 4 using pure Python

.. warning::
    These core transforms will produce aliasing when used without overlap.
    Please use :py:mod:`mdct` unless you know what this means.

"""

from __future__ import division

import numpy

__all__ = [
    'mdct', 'imdct',
    'mdst', 'imdst',
    'cmdct', 'icmdct',
    'mclt', 'imclt',
]


def mdct(x, odd=True):
    """ Calculate modified discrete cosine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return trans(x, func=numpy.cos, odd=odd) * numpy.sqrt(2)


def imdct(X, odd=True):
    """ Calculate inverse modified discrete cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return itrans(X, func=numpy.cos, odd=odd) * numpy.sqrt(2)


def mdst(x, odd=True):
    """ Calculate modified discrete sine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return trans(x, func=numpy.sin, odd=odd) * numpy.sqrt(2)


def imdst(X, odd=True):
    """ Calculate inverse modified discrete sine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return itrans(X, func=numpy.sin, odd=odd) * numpy.sqrt(2)


def cmdct(x, odd=True):
    """ Calculate complex modified discrete cosine transform of input
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return trans(x, func=lambda x: numpy.cos(x) - 1j * numpy.sin(x), odd=odd)


def icmdct(X, odd=True):
    """ Calculate inverse complex modified discrete cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    return itrans(X, func=lambda x: numpy.cos(x) + 1j * numpy.sin(x), odd=odd)


mclt = cmdct
imclt = icmdct


def trans(x, func, odd=True):
    """ Calculate modified discrete sine/cosine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    func : callable
        The transform kernel function
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x) // 2
    if odd:
        outlen = N
        offset = 0.5
    else:
        outlen = N + 1
        offset = 0.0

    X = numpy.zeros(outlen, dtype=numpy.complex)
    n = numpy.arange(len(x))

    for k in range(len(X)):
        X[k] = numpy.sum(
            x * func(
                (numpy.pi / N) * (
                    n + 0.5 + N / 2
                ) * (
                    k + offset
                )
            )
        )

    if not odd:
        X[0] *= numpy.sqrt(0.5)
        X[-1] *= numpy.sqrt(0.5)

    return X * numpy.sqrt(1 / N)


def itrans(X, func, odd=True):
    """ Calculate inverse modified discrete sine/cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    func : callable
        The transform kernel function
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.

    Returns
    -------
    out : array_like
        The output signal

    """
    if not odd and len(X) % 2 == 0:
        raise ValueError(
            "Even inverse CMDCT requires an odd number "
            "of coefficients"
        )

    X = X.copy()

    if odd:
        N = len(X)
        offset = 0.5
    else:
        N = len(X) - 1
        offset = 0.0

        X[0] *= numpy.sqrt(0.5)
        X[-1] *= numpy.sqrt(0.5)

    x = numpy.zeros(N * 2, dtype=numpy.complex)
    k = numpy.arange(len(X))

    for n in range(len(x)):
        x[n] = numpy.sum(
            X * func(
                (numpy.pi / N) * (
                    n + 0.5 + N / 2
                ) * (
                    k + offset
                )
            )
        )

    return numpy.real(x) * numpy.sqrt(1 / N)
