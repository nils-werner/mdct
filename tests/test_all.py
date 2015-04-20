import pytest
import itertools
import numpy
import scipy.signal
import mdct
import mdct.slow
import stft


fast_functions = [
    (mdct.fast.mdct, mdct.fast.imdct),
    (mdct.fast.mdst, mdct.fast.imdst),
    (mdct.fast.cmdct, mdct.fast.icmdct)
]

slow_functions = [
    (mdct.slow.mdct, mdct.slow.imdct),
    (mdct.slow.mdst, mdct.slow.imdst),
    (mdct.slow.cmdct, mdct.slow.icmdct)
]


corresponding_functions = list(zip(fast_functions, slow_functions))

any_functions = fast_functions + slow_functions

all_functions = list(itertools.chain.from_iterable(
    [
        list(itertools.product(*zip(*item)))
        for item in corresponding_functions
    ]
))

cross_functions = [
    item for item in all_functions
    if item not in itertools.chain.from_iterable(corresponding_functions)
]

forward_functions = [(a, b) for (a, _), (b, _) in corresponding_functions]
backward_functions = [(a, b) for (_, a), (_, b) in corresponding_functions]


@pytest.fixture
def length():
    return 5


@pytest.fixture
def N(length):
    # must be multiple of 1024 b/c of stft
    return length * 1024


@pytest.fixture(params=(100., 1000.))
def sig(N, request):
    return numpy.sin(numpy.arange(N) / 44100. * request.param * 2 * numpy.pi)


@pytest.fixture
def spectrum(N, length):
    return numpy.random.rand(N // (length * 2), length * 2 + 1)


@pytest.fixture(params=(
    mdct.fast,
    mdct.slow,
))
def module(request):
    return request.param


@pytest.fixture(params=fast_functions)
def fast_function(request):
    """ This fixture returns all fast transforms and its fast inverse

    """
    return request.param


@pytest.fixture(params=slow_functions)
def slow_function(request):
    """ This fixture returns all slow transforms and its slow inverse

    """
    return request.param


@pytest.fixture(params=any_functions)
def any_function(request):
    """ This fixture combines all transforms with its module-internal
    inverse, (to test fast and slow in isolation)

    """
    return request.param


@pytest.fixture(params=all_functions)
def all_function(request):
    """ This fixture combines all possible combinations of transform
    and its inverse, across modules (to test fast vs slow inversability)

    """
    return request.param


@pytest.fixture(params=cross_functions)
def cross_function(request):
    """ This fixture combines all combinations of transform
    and its inverse from the other modules (to test fast-slow inversability)

    """
    return request.param


@pytest.fixture(params=forward_functions)
def forward_function(request):
    """ This fixture combines all forward transforms and
    their correspondence from the other modules (to test fast-slow equality)

    """
    return request.param


@pytest.fixture(params=backward_functions)
def backward_function(request):
    """ This fixture combines all backward transforms and
    their correspondence from the other modules (to test fast-slow equality)

    """
    return request.param


def test_halving(sig, module):
    assert len(module.transforms.mdct(sig)) == len(sig) // 2
    assert len(module.transforms.mdst(sig)) == len(sig) // 2
    assert len(module.transforms.cmdct(sig)) == len(sig) // 2


def test_outtypes(sig, module):
    assert numpy.all(numpy.isreal(module.transforms.mdct(sig)))
    assert numpy.all(numpy.isreal(module.transforms.mdst(sig)))
    assert numpy.all(numpy.iscomplex(module.transforms.cmdct(sig)))


def test_forward_equality(sig, forward_function):
    spec = forward_function[0](sig)
    spec2 = forward_function[1](sig)

    assert spec.shape == spec2.shape
    assert numpy.allclose(spec, spec2)


def test_backward_equality(spectrum, backward_function):
    sig = backward_function[0](spectrum)
    sig2 = backward_function[1](spectrum)

    assert sig.shape == sig2.shape
    assert numpy.allclose(sig, sig2)


def test_inverse(sig, cross_function):
    spec = cross_function[0](sig)
    outsig = cross_function[1](spec)

    assert numpy.all(numpy.isreal(outsig))
    assert len(outsig) == len(sig)
    assert numpy.allclose(outsig, sig)
