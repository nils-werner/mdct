import pytest
import numpy
import mdct.windows


def test_kbd():
    M = 100
    w = mdct.windows.kaiser_derived(M, beta=4.)

    assert numpy.allclose(w[:M//2] ** 2 + w[-M//2:] ** 2, 1.)

    with pytest.raises(ValueError):
        mdct.windows.kaiser_derived(M + 1, beta=4.)

    assert numpy.allclose(
        mdct.windows.kaiser_derived(2, beta=numpy.pi/2)[:1],
        [numpy.sqrt(2)/2])

    assert numpy.allclose(
        mdct.windows.kaiser_derived(4, beta=numpy.pi/2)[:2],
        [0.518562710536, 0.855039598640])

    assert numpy.allclose(
        mdct.windows.kaiser_derived(6, beta=numpy.pi/2)[:3],
        [0.436168993154, 0.707106781187, 0.899864772847])
