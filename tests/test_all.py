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

fast_unlapped_functions = [
    (mdct.fast.transforms.cmdct, mdct.fast.transforms.icmdct),
]

slow_unlapped_functions = [
    (mdct.slow.transforms.cmdct, mdct.slow.transforms.icmdct),
]


def correspondences(a, b):
    """ Returns

     - fast.forwards->slow.forwards
     - fast.backwards->slow.backwards

    """
    return list(zip(a, b))


def merge(a, b):
    """ Returns

     - slow.forwards->slow.backwards
     - fast.forwards->fast.backwards

    """
    return a + b


def combinations(a, b):
    """ Returns all combinations of

     - fast.forwards->fast.backwards
     - fast.forwards->slow.backwards
     - slow.forwards->fast.backwards
     - slow.forwards->slow.backwards

    """
    return list(itertools.chain.from_iterable(
        [
            list(itertools.product(*zip(*item)))
            for item in correspondences(a, b)
        ]
    ))


def crossings(a, b):
    """ Returns

     - fast.forwards->slow.backwards
     - slow.forwards->fast.backwards

     but never

     - slow.forwards->slow.backwards
     - fast.forwards->fast.backwards

    """
    return [
        item for item in combinations(a, b)
        if item not in itertools.chain.from_iterable(correspondences(a, b))
    ]


def forward(a, b):
    """ Returns

     - fast.forwards->slow.forwards

    """
    return [(x, y) for (x, _), (y, _) in correspondences(a, b)]


def backward(a, b):
    """ Returns

     - fast.backwards->slow.backwards

    """
    return [(x, y) for (_, x), (_, y) in correspondences(a, b)]


corresponding_functions = correspondences(fast_functions, slow_functions)
any_functions = merge(fast_functions, slow_functions)
all_functions = combinations(fast_functions, slow_functions)
cross_functions = crossings(fast_functions, slow_functions)
forward_functions = forward(fast_functions, slow_functions)
backward_functions = backward(fast_functions, slow_functions)

unlapped_functions = combinations(
    fast_unlapped_functions, slow_unlapped_functions
)

forward_unlapped_functions = \
    forward(fast_unlapped_functions, slow_unlapped_functions)

backward_unlapped_functions = \
    backward(fast_unlapped_functions, slow_unlapped_functions)


def test_halving(sig, module):
    #
    # Test if the output is half the size of the input
    #
    assert len(module.transforms.mdct(sig)) == len(sig) // 2
    assert len(module.transforms.mdst(sig)) == len(sig) // 2
    assert len(module.transforms.cmdct(sig)) == len(sig) // 2


def test_outtypes(sig, module):
    assert numpy.all(numpy.isreal(module.transforms.mdct(sig)))
    assert numpy.all(numpy.isreal(module.transforms.mdst(sig)))
    assert numpy.any(numpy.iscomplex(module.transforms.cmdct(sig)))
    assert numpy.all(numpy.isreal(module.transforms.icmdct(sig)))


@pytest.mark.parametrize("function", forward_functions)
def test_forward_equality(sig, function):
    #
    # Test if slow and fast transforms are equal. Tests all with lapping.
    #
    spec = function[0](sig)
    spec2 = function[1](sig)

    assert spec.shape == spec2.shape
    assert numpy.allclose(spec, spec2)


@pytest.mark.parametrize("function", backward_functions)
def test_backward_equality(spectrum, function):
    #
    # Test if slow and fast inverse transforms are equal.
    # Tests all with lapping.
    #
    sig = function[0](spectrum)
    sig2 = function[1](spectrum)

    assert sig.shape == sig2.shape
    assert numpy.allclose(sig, sig2)


@pytest.mark.parametrize("function", all_functions)
def test_inverse(sig, function):
    spec = function[0](sig)
    outsig = function[1](spec)
    #
    # Test if combinations slow-slow, slow-fast, fast-fast, fast-slow are all
    # perfect reconstructing. Tests all with lapping.
    #

    assert numpy.all(numpy.isreal(outsig))
    assert len(outsig) == len(sig)
    assert numpy.allclose(outsig, sig)


@pytest.mark.parametrize("function", unlapped_functions)
def test_unlapped_inverse(sig, function):
    #
    # Test if combinations slow-slow, slow-fast, fast-fast, fast-slow are all
    # perfect reconstructing. Tests only CMDCT without lapping.
    #
    spec = function[0](sig)
    outsig = function[1](spec)

    assert len(outsig) == len(sig)
    assert numpy.allclose(outsig, sig)


@pytest.mark.parametrize("function", forward_unlapped_functions)
def test_unlapped_equality(sig, function):
    #
    # Test if slow and fast unlapped transforms are equal. Tests only
    # CDMCT without lapping.
    #
    outsig = function[0](sig)
    outsig2 = function[1](sig)

    assert outsig.shape == outsig2.shape
    assert numpy.allclose(outsig, outsig2)


@pytest.mark.parametrize("function", backward_unlapped_functions)
def test_unlapped_backwards_equality(sig, function):
    #
    # Test if slow and fast unlapped inverse transforms are equal. Tests only
    # CDMCT without lapping.
    #
    outsig = function[0](sig)
    outsig2 = function[1](sig)

    assert outsig.shape == outsig2.shape
    assert numpy.allclose(outsig, outsig2)
