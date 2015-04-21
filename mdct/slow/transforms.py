""" Module for calculating DCT type 4 using pure Python

"""

from __future__ import division

import numpy

__all__ = [
    'mdct', 'imdct',
    'mdst', 'imdst',
    'cmdct', 'icmdct',
    'mclt', 'imclt',
]


def mdct(x):
    """ Calculate modified discrete cosine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    return trans(x, func=numpy.cos) * numpy.sqrt(2)


def imdct(X):
    """ Calculate inverse modified discrete cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    return itrans(X, func=numpy.cos) * numpy.sqrt(2)


def mdst(x):
    """ Calculate modified discrete sine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    return trans(x, func=numpy.sin) * numpy.sqrt(2)


def imdst(X):
    """ Calculate inverse modified discrete sine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    return itrans(X, func=numpy.sin) * numpy.sqrt(2)


def cmdct(x):
    """ Calculate complex modified discrete cosine transform of input
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    # return trans(x, func=lambda x: numpy.cos(x) - 1j * numpy.sin(x))
    N = len(x) // 2
    X = numpy.zeros(N, dtype=numpy.complex)

    for k in range(len(X)):
        X[k] = numpy.sum(
            x * (
                numpy.cos(
                    (numpy.pi / N) * (
                        numpy.arange(2 * N) + 0.5 + N / 2
                    ) * (
                        k + 0.5
                    )
                ) - 1j * numpy.sin(
                    (numpy.pi / N) * (
                        numpy.arange(2 * N) + 0.5 + N / 2
                    ) * (
                        k + 0.5
                    )
                )
            )
        )

    return X * numpy.sqrt(1 / N)


def icmdct(X):
    """ Calculate inverse complex modified discrete cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    # return itrans(X, func=lambda x: numpy.cos(x) + 1j * numpy.sin(x))
    N = len(X)
    x = numpy.zeros(N * 2, dtype=numpy.complex)

    for n in range(len(x)):
        x[n] = numpy.sum(
            X * (
                numpy.cos(
                    (numpy.pi / N) * (
                        n + 0.5 + N / 2
                    ) * (
                        numpy.arange(N) + 0.5
                    )
                ) + 1j * numpy.sin(
                    (numpy.pi / N) * (
                        n + 0.5 + N / 2
                    ) * (
                        numpy.arange(N) + 0.5
                    )
                )
            )
        )

    return x * numpy.sqrt(1 / N)


mclt = cmdct
imclt = icmdct


def trans(x, func):
    """ Calculate modified discrete sine/cosine transform of input signal in an
    inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    func : callable
        The transform kernel function

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x) // 2
    X = numpy.zeros(N)

    for k in range(len(X)):
        X[k] = numpy.sum(
            x * func(
                (numpy.pi / N) * (
                    numpy.arange(2 * N) + 0.5 + N / 2
                ) * (
                    k + 0.5
                )
            )
        )

    return X * numpy.sqrt(1 / N)


def itrans(X, func):
    """ Calculate inverse modified discrete sine/cosine transform of input
    signal in an inefficient pure-Python method.

    Use only for testing.

    Parameters
    ----------
    X : array_like
        The input signal
    func : callable
        The transform kernel function

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(X)
    x = numpy.zeros(N * 2)

    for n in range(len(x)):
        x[n] = numpy.sum(
            X * func(
                (numpy.pi / N) * (
                    n + 0.5 + N / 2
                ) * (
                    numpy.arange(N) + 0.5
                )
            )
        )

    return x * numpy.sqrt(1 / N)
