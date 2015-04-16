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


@pytest.fixture(params=(
    (mdct.mdct, mdct.imdct),
    (mdct.mdst, mdct.imdst),
    (mdct.cmdct, mdct.icmdct),
))
def function(request):
    return request.param


def test_halving(sig):
    assert len(mdct.transforms.mdct(sig)) == len(sig) // 2
    assert len(mdct.transforms.mdst(sig)) == len(sig) // 2
    assert len(mdct.transforms.cmdct(sig)) == len(sig) // 2


def test_inverse(sig, function):
    spec = function[0](sig)
    outsig = function[1](spec)

    assert len(outsig) == len(sig)
    assert numpy.allclose(outsig, sig)
