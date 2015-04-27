import numpy
import pytest
import mdct


def test_weird_backtransform(sig, framelength):
    spec = numpy.array(mdct.mdct(
        sig,
        framelength=framelength,
    ))

    outsig = mdct.imdct(
        spec,
        framelength=framelength,
    )

    assert sig.shape == outsig.shape
    assert numpy.allclose(sig, outsig)
