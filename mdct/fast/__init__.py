""" Module for calculating lapped MDCT using FFT

.. warning::
    Functions defined in this module are exposed using the :py:mod:`mdct`
    module itself, please do not use this module directly.

"""

import functools

import stft
from . import transforms as transforms_default

__all__ = [
    'mdct', 'imdct',
    'mdst', 'imdst',
    'cmdct', 'icmdct',
    'mclt', 'imclt',
]


def mdct(
    x,
    odd=True,
    transforms=None,
    **kwargs
):
    """ Calculate lapped MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
        Additional keyword arguments passed to :code:`stft.spectrogram`

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.mdct : MDCT

    """
    if transforms is None:
        transforms = transforms_default

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
    transforms=None,
    **kwargs
):
    """ Calculate lapped inverse MDCT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
        Additional keyword arguments passed to :code:`stft.spectrogram`

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.imdct : inverse MDCT

    """
    if transforms is None:
        transforms = transforms_default

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
    transforms=None,
    **kwargs
):
    """ Calculate lapped MDST of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
        Additional keyword arguments passed to :code:`stft.spectrogram`

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.mdst : MDST

    """
    if transforms is None:
        transforms = transforms_default

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
    transforms=None,
    **kwargs
):
    """ Calculate lapped inverse MDST of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
        Additional keyword arguments passed to :code:`stft.spectrogram`

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.imdst : inverse MDST

    """
    if transforms is None:
        transforms = transforms_default

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
    transforms=None,
    **kwargs
):
    """ Calculate lapped complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
    **kwargs, optional
        Additional keyword arguments passed to :code:`stft.spectrogram`

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.cmdct : complex MDCT

    """
    if transforms is None:
        transforms = transforms_default

    return stft.spectrogram(
        x,
        transform=functools.partial(transforms.cmdct, odd=odd),
        halved=False,
        **kwargs
    )


def icmdct(
    X,
    odd=True,
    transforms=None,
    **kwargs
):
    """ Calculate lapped inverse complex MDCT/MCLT of input signal

    Parameters
    ----------
    x : array_like
        The input signal
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
        Additional keyword arguments passed to :code:`stft.spectrogram`

    Returns
    -------
    out : array_like
        The output signal

    See Also
    --------
    mdct.fast.transforms.icmdct : inverse complex MDCT

    """
    if transforms is None:
        transforms = transforms_default

    return stft.ispectrogram(
        X,
        transform=functools.partial(transforms.icmdct, odd=odd),
        halved=False,
        **kwargs
    )

mclt = cmdct
imclt = icmdct
