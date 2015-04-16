MDCT documentation
==================

This toolkit implements several related transforms and their inverses:

 - Modulated Complex Lapped Transform (MCLT) aka Complex Modified Discrete Cosine Transform (CMDCT)
 - Modified Discrete Cosine Transform (MDCT)
 - Modified Discrete Sine Transform (MDST)

All transforms are implemented as an un-windowed standalone transform as well as
their lapped counterpart, along with windowing and overlap-add reconstruction.

The implementation used here is based on the FFT wrapper published in [Borsi].

.. toctree::
   :maxdepth: 2
   :hidden:

   mdct
   mdct.transforms
   mdct.lapped
   mdct.windows

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [Borsi] Marina Bosi, Richard E. Goldberg and Leonardo Chiariglione,
   "Introduction to Digital Audio Coding and Standards", Kluwer Academic Publishers, 01 December, 2002.
