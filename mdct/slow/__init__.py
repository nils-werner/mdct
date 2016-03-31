""" Module for calculating lapped MDCT using pure Python

.. warning::
    This module is very slow and only meant for testing. Use :py:mod:`mdct`
    instead.

"""

import functools
from . import transforms
from .. import fast

__all__ = [
    'mdct', 'imdct',
    'mdst', 'imdst',
    'cmdct', 'icmdct',
    'mclt', 'imclt',
]

mdct = functools.partial(fast.mdct, transforms=transforms)
imdct = functools.partial(fast.imdct, transforms=transforms)
mdst = functools.partial(fast.mdst, transforms=transforms)
imdst = functools.partial(fast.imdst, transforms=transforms)
cmdct = functools.partial(fast.cmdct, transforms=transforms)
icmdct = functools.partial(fast.icmdct, transforms=transforms)

mclt = cmdct
imclt = icmdct
