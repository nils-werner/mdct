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
    framelength=1024,
    window=None,
    odd=True,
):
    """ Calculate lapped MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.spectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.mdct : MDCT

    """
    if not odd:
        return stft.spectrogram(
            x,
            halved=False,
            framelength=framelength,
            window=window,
            transform=[
                functools.partial(transforms.mdct, odd=False),
                functools.partial(transforms.mdst, odd=False),
            ]
        )
    else:
        return stft.spectrogram(
            x,
            halved=False,
            framelength=framelength,
            window=window,
            transform=transforms.mdct,
        )


def imdct(
    X,
    framelength=1024,
    window=None,
    odd=True,
):
    """ Calculate lapped inverse MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.ispectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.imdct : inverse MDCT

    """
    if not odd:
        return stft.ispectrogram(
            X,
            halved=False,
            framelength=framelength,
            window=window,
            transform=[
                functools.partial(transforms.imdct, odd=False),
                functools.partial(transforms.imdst, odd=False),
            ]
        )
    else:
        return stft.ispectrogram(
            X,
            halved=False,
            framelength=framelength,
            window=window,
            transform=transforms.imdct,
        )


def mdst(
    x,
    framelength=1024,
    window=None,
    odd=True,
):
    """ Calculate lapped MDST of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.spectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.mdst : MDST

    """
    if not odd:
        return stft.spectrogram(
            x,
            halved=False,
            framelength=framelength,
            window=window,
            transform=[
                functools.partial(transforms.mdst, odd=False),
                functools.partial(transforms.mdct, odd=False),
            ]
        )
    else:
        return stft.spectrogram(
            x,
            halved=False,
            framelength=framelength,
            window=window,
            transform=transforms.mdst,
        )


def imdst(
    X,
    framelength=1024,
    window=None,
    odd=True,
):
    """ Calculate lapped inverse MDST of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.ispectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.imdst : inverse MDST

    """
    if not odd:
        return stft.ispectrogram(
            X,
            halved=False,
            framelength=framelength,
            window=window,
            transform=[
                functools.partial(transforms.imdst, odd=False),
                functools.partial(transforms.imdct, odd=False),
            ]
        )
    else:
        return stft.ispectrogram(
            X,
            halved=False,
            framelength=framelength,
            window=window,
            transform=transforms.imdst,
        )


def cmdct(
    x,
    framelength=1024,
    window=None,
    odd=True,
):
    """ Calculate lapped complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.spectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

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
        halved=False,
        framelength=framelength,
        window=window,
        transform=functools.partial(transforms.cmdct, odd=odd),
    )


def icmdct(
    X,
    framelength=1024,
    window=None,
    odd=True,
):
    """ Calculate lapped inverse complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    framelength : int
        Framesize for :code:`stft.ispectrogram`. Defaults to 1024.
    window : array_like
        Window for :code:`stft.ispectrogram`.
        Defaults to :code:`scipy.signal.cosine`.

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
        halved=False,
        framelength=framelength,
        window=window,
        transform=functools.partial(transforms.icmdct, odd=odd),
    )

mclt = cmdct
imclt = icmdct
