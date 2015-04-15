import numpy
import scipy
import stft


def mdct(x):
    N = len(x)
    n0 = (N / 2 + 1) / 2

    x = x * numpy.exp(-1j * 2 * numpy.pi * numpy.arange(N) / 2 / N)
    X = scipy.fftpack.fft(x)

    return numpy.real(
        X[:N/2] * numpy.exp(
            -1j * 2 * numpy.pi * n0 * (numpy.arange(N/2) + 0.5) / N
        )
    )


def imdct(X):
    N = 2 * len(X)
    n0 = (N / 2 + 1) / 2

    Y = numpy.zeros(N)

    Y[:N/2] = X
    Y[N/2:] = numpy.real(-1 * numpy.flipud(X))

    Y *= numpy.abs(numpy.exp(1j * 2 * numpy.pi * numpy.arange(N) * n0 / N))
    y = scipy.fftpack.ifft(Y)

    return 2 * numpy.real(
        y * numpy.exp(1j * 2 * numpy.pi * (numpy.arange(N) + n0) / 2 / N)
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
