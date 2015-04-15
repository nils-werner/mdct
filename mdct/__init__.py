from __future__ import division
import numpy
import scipy
import stft


def mdct(x):
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


def spectrogram(x):
    return stft.spectrogram(
        x,
        halved=False,
        transform=mdct
    )


def ispectrogram(X):
    return stft.ispectrogram(
        X,
        halved=False,
        transform=imdct
    )
