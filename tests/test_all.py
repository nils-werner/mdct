import pytest
import numpy
import scipy.signal
import mdct
import stft


@pytest.fixture
def N():
    # Roughly 44100 * 10, must be multiple of 1024 b/c of stft
    return 430 * 1024


@pytest.fixture(params=(100., 1000.))
def sig(N, request):
    return numpy.sin(numpy.arange(N) / request.param)


def test_lengths(sig):
    assert len(mdct.mdct(sig)) == len(sig) // 2
    assert len(mdct.ispectrogram(mdct.spectrogram(sig))) == len(sig)


def test_inverse(sig):
    spec = mdct.spectrogram(sig)
    outsig = mdct.ispectrogram(spec)

    assert numpy.allclose(outsig, sig)
