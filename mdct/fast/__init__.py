""" Module for calculating lapped MDCT

.. note::
    Functions defined in this module are exposed using the :mod:`mdct` module
    itself, meaning :code:`mdct.mdct == mdct.fast.mdct` etc.

"""

import functools

import stft
from . import transforms

__all__ = [
    'mdct', 'imdct',
    'mdst', 'imdst',
    'cmdct', 'icmdct',
    'mclt', 'imclt',
]


def mdct(
    x,
    odd=True,
    **kwargs
):
    """ Calculate lapped MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.mdct : MDCT

    """
    kwargs.setdefault('framelength', 1024)

    if not odd:
        return stft.spectrogram(
            x,
            transform=[
                functools.partial(transforms.mdct, odd=False),
                functools.partial(transforms.mdst, odd=False),
            ],
            halved=False,
            **kwargs
        )
    else:
        return stft.spectrogram(
            x,
            transform=transforms.mdct,
            halved=False,
            **kwargs
        )


def imdct(
    X,
    odd=True,
    **kwargs
):
    """ Calculate lapped inverse MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.imdct : inverse MDCT

    """
    kwargs.setdefault('framelength', 1024)

    if not odd:
        return stft.ispectrogram(
            X,
            transform=[
                functools.partial(transforms.imdct, odd=False),
                functools.partial(transforms.imdst, odd=False),
            ],
            halved=False,
            **kwargs
        )
    else:
        return stft.ispectrogram(
            X,
            transform=transforms.imdct,
            halved=False,
            **kwargs
        )


def mdst(
    x,
    odd=True,
    **kwargs
):
    """ Calculate lapped MDST of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.mdst : MDST

    """
    kwargs.setdefault('framelength', 1024)

    if not odd:
        return stft.spectrogram(
            x,
            transform=[
                functools.partial(transforms.mdst, odd=False),
                functools.partial(transforms.mdct, odd=False),
            ],
            halved=False,
            **kwargs
        )
    else:
        return stft.spectrogram(
            x,
            transform=transforms.mdst,
            halved=False,
            **kwargs
        )


def imdst(
    X,
    odd=True,
    **kwargs
):
    """ Calculate lapped inverse MDST of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.imdst : inverse MDST

    """
    kwargs.setdefault('framelength', 1024)

    if not odd:
        return stft.ispectrogram(
            X,
            transform=[
                functools.partial(transforms.imdst, odd=False),
                functools.partial(transforms.imdct, odd=False),
            ],
            halved=False,
            **kwargs
        )
    else:
        return stft.ispectrogram(
            X,
            transform=transforms.imdst,
            halved=False,
            **kwargs
        )


def cmdct(
    x,
    odd=True,
    **kwargs
):
    """ Calculate lapped complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.cmdct : complex MDCT

    """
    return stft.spectrogram(
        x,
        transform=functools.partial(transforms.cmdct, odd=odd),
        halved=False,
        **kwargs
    )


def icmdct(
    X,
    odd=True,
    **kwargs
):
    """ Calculate lapped inverse complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.icmdct : inverse complex MDCT

    """
    return stft.ispectrogram(
        X,
        transform=functools.partial(transforms.icmdct, odd=odd),
        halved=False,
        **kwargs
    )

mclt = cmdct
imclt = icmdct
