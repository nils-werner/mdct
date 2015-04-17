import pytest
import numpy
import mdct.windows


def test_kbd():
    M = 100
    w = mdct.windows.kaiser_derived(M)

    assert numpy.allclose(w[:M//2] ** 2 + w[-M//2:] ** 2, 1.)

    with pytest.raises(ValueError):
        mdct.windows.kaiser_derived(51)
