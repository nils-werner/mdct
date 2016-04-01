MDCT
====

This toolkit implements several related transforms and their inverses:

 - **Modified Discrete Cosine Transform (MDCT)**
 - Modified Discrete Sine Transform (MDST)
 - Modulated Complex Lapped Transform (MCLT) aka Complex Modified Discrete Cosine Transform (CMDCT)

All transforms are implemented as

 - the complete lapped transform, along with windowing and time domain aliasing cancellation (TDAC) reconstruction and
 - the core un-windowed standalone transform.

All transforms are implemeted in

 - :py:mod:`mdct.fast`,a fast, FFT-based method (for actual use), see [Bosi]
 - :py:mod:`mdct.slow`, a slow, pure-Python fashion (for testing) and

Usage
-----

.. warning::
    :py:mod:`mdct.fast` is exposed as :py:mod:`mdct`. Please use this module directly.

.. code-block:: python

    import mdct
    spec = mdct.mdct(signal)
    output = mdct.imdct(spec)

.. toctree::
   :hidden:

   self
   modules
   internal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [Bosi] Marina Bosi, Richard E. Goldberg and Leonardo Chiariglione,
   "Introduction to Digital Audio Coding and Standards", Kluwer Academic Publishers, 01 December, 2002.
