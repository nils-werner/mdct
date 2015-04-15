from __future__ import division
import numpy
import scipy
import stft

from . import windows

__all__ = ['mdct', 'imdct']


def mdct(x):
    """ Calculate MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    N = len(x)
    n0 = (N / 2 + 1) / 2

    X = scipy.fftpack.fft(
        x * numpy.exp(-1j * 2 * numpy.pi * numpy.arange(N) / 2 / N)
    )

    return numpy.real(
        X[:N/2] * numpy.exp(
            -1j * 2 * numpy.pi * n0 * (numpy.arange(N / 2) + 0.5) / N
        )
    )


def imdct(X):
    """ Calculate inverse MDCT of input signal

    Parameters
    ----------
    X : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    """
    N = 2 * len(X)
    n0 = (N / 2 + 1) / 2

    Y = numpy.zeros(N, dtype=X.dtype)

    Y[:N/2] = X
    Y[N/2:] = -1 * X[::-1]

    y = scipy.fftpack.ifft(
        Y * numpy.exp(1j * 2 * numpy.pi * numpy.arange(N) * n0 / N)
    )

    return 2 * numpy.real(
        y * numpy.exp(
            1j * 2 * numpy.pi * (numpy.arange(N) + n0) / 2 / N
        )
    )


def spectrogram(
    x,
    framelength=1024,
    window=None
):
    """ Calculate windowed MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.spectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct : MDCT

    """
    return stft.spectrogram(
        x,
        halved=False,
        framelength=framelength,
        window=window,
        transform=mdct
    )


def ispectrogram(
    X,
    framelength=1024,
    window=None
):
    """ Calculate lapped inverse MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.ispectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    imdct : inverse MDCT

    """
    return stft.ispectrogram(
        X,
        halved=False,
        framelength=framelength,
        window=window,
        transform=imdct
    )
