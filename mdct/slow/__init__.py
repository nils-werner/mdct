""" Module for calculating lapped MDCT

.. note::
    This module is very slow and only meant for testing

"""

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
    window=None
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
    mdct.slow.transforms.mdct : MDCT

    """
    return stft.spectrogram(
        x,
        halved=False,
        framelength=framelength,
        window=window,
        transform=transforms.mdct
    )


def imdct(
    X,
    framelength=1024,
    window=None
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
    mdct.slow.transforms.imdct : inverse MDCT

    """
    return stft.ispectrogram(
        X,
        halved=False,
        framelength=framelength,
        window=window,
        transform=transforms.imdct
    )


def mdst(
    x,
    framelength=1024,
    window=None
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
    mdct.slow.transforms.mdst : MDST

    """
    return stft.spectrogram(
        x,
        halved=False,
        framelength=framelength,
        window=window,
        transform=transforms.mdst
    )


def imdst(
    X,
    framelength=1024,
    window=None
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
    mdct.slow.transforms.imdst : inverse MDST

    """
    return stft.ispectrogram(
        X,
        halved=False,
        framelength=framelength,
        window=window,
        transform=transforms.imdst
    )


def cmdct(
    x,
    framelength=1024,
    window=None
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
    mdct.slow.transforms.cmdct : complex MDCT

    """
    return stft.spectrogram(
        x,
        halved=False,
        framelength=framelength,
        window=window,
        transform=transforms.cmdct
    )


def icmdct(
    X,
    framelength=1024,
    window=None
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
    mdct.slow.transforms.icmdct : inverse complex MDCT

    """
    return stft.ispectrogram(
        X,
        halved=False,
        framelength=framelength,
        window=window,
        transform=transforms.icmdct
    )

mclt = cmdct
imclt = icmdct
