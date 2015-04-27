import numpy
import scipy
import pytest
import mdct
import mdct.slow
import random as rand
import functools


@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)


@pytest.fixture
def length():
    return 5


@pytest.fixture
def N(length):
    # must be multiple of 1024 b/c of stft
    return length * 1024


@pytest.fixture
def sig(N, random):
    return numpy.random.rand(N)


@pytest.fixture(params=(1024,))
def framelength(request):
    return request.param


@pytest.fixture
def backsig(N, random, odd):
    if odd:
        return numpy.random.rand(N)
    else:
        return numpy.random.rand(N + 1)


@pytest.fixture
def spectrum(N, random, length, odd):
    if odd:
        return numpy.random.rand(N // (length * 2), length * 2 + 1)
    else:
        return numpy.random.rand(N // (length * 2) + 1, length * 2 + 1)


@pytest.fixture(params=(
    mdct.fast,
    mdct.slow,
))
def module(request):
    return request.param


@pytest.fixture(params=(True, False))
def odd(request):
    return request.param


@pytest.fixture(params=(
    scipy.signal.cosine,
    functools.partial(mdct.windows.kaiser_derived, beta=4.)
))
def window(request):
    return request.param
