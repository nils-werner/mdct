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
        The signal to be transformed. May be a 1D vector for single channel or
        a 2D matrix for multi channel data. In case of a mono signal, the data
        is must be a 1D vector of length :code:`samples`. In case of a multi
        channel signal, the data must be in the shape of :code:`samples x
        channels`.
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    framelength : int
        The signal frame length. Defaults to :code:`2048`.
    hopsize : int
        The signal frame hopsize. Defaults to :code:`None`. Setting this
        value will override :code:`overlap`.
    overlap : int
        The signal frame overlap coefficient. Value :code:`x` means
        :code:`1/x` overlap. Defaults to :code:`2`. Note that anything but
        :code:`2` will result in a filterbank without perfect reconstruction.
    centered : boolean
        Pad input signal so that the first and last window are centered around
        the beginning of the signal. Defaults to :code:`True`.
        Disabling this will result in aliasing
        in the first and last half-frame.
    window : callable, array_like
        Window to be used for deringing. Can be :code:`False` to disable
        windowing. Defaults to :code:`scipy.signal.cosine`.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
    padding : int
        Zero-pad signal with x times the number of samples.
        Defaults to :code:`0`.
    save_settings : boolean
        Save settings used here in attribute :code:`out.stft_settings` so that
        :func:`ispectrogram` can infer these settings without the developer
        having to pass them again.

    Returns
    -------
    out : array_like
        The signal (or matrix of signals). In case of a mono output signal, the
        data is formatted as a 1D vector of length :code:`samples`. In case of
        a multi channel output signal, the data is formatted as :code:`samples
        x channels`.

    See Also
    --------
    mdct.fast.transforms.mdct : MDCT

    """
    if transforms is None:
        transforms = transforms_default

    kwargs.setdefault('framelength', 2048)

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
        The spectrogram to be inverted. May be a 2D matrix for single channel
        or a 3D tensor for multi channel data. In case of a mono signal, the
        data must be in the shape of :code:`bins x frames`. In case of a multi
        channel signal, the data must be in the shape of :code:`bins x frames x
        channels`.
    odd : boolean, optional
        Switch to oddly stacked transform. Defaults to :code:`True`.
    framelength : int
        The signal frame length. Defaults to infer from data.
    hopsize : int
        The signal frame hopsize. Defaults to infer from data. Setting this
        value will override :code:`overlap`.
    overlap : int
        The signal frame overlap coefficient. Value :code:`x` means
        :code:`1/x` overlap. Defaults to infer from data. Note that anything
        but :code:`2` will result in a filterbank without perfect
        reconstruction.
    centered : boolean
        Pad input signal so that the first and last window are centered around
        the beginning of the signal. Defaults to to infer from data.
        The first and last half-frame will have aliasing, so using
        centering during forward MDCT is recommended.
    window : callable, array_like
        Window to be used for deringing. Can be :code:`False` to disable
        windowing. Defaults to to infer from data.
    halved : boolean
        Switch to reconstruct the other halve of the spectrum if the forward
        transform has been truncated. Defaults to to infer from data.
    transforms : module, optional
        Module reference to core transforms. Mostly used to replace
        fast with slow core transforms, for testing. Defaults to
        :mod:`mdct.fast`
    padding : int
        Zero-pad signal with x times the number of samples. Defaults to infer
        from data.
    outlength : int
        Crop output signal to length. Useful when input length of spectrogram
        did not fit into framelength and input data had to be padded. Not
        setting this value will disable cropping, the output data may be
        longer than expected.

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

    kwargs.setdefault('framelength', 2048)

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

    kwargs.setdefault('framelength', 2048)

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

    kwargs.setdefault('framelength', 2048)

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
