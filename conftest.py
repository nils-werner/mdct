import numpy
import pytest
import mdct
import mdct.slow
import random as rand


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


@pytest.fixture
def spectrum(N, random, length):
    return numpy.random.rand(N // (length * 2), length * 2 + 1)


@pytest.fixture(params=(
    mdct.fast,
    mdct.slow,
))
def module(request):
    return request.param
