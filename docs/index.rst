MDCT documentation
==================

Contents:

.. toctree::
   :maxdepth: 2

   mdct

Wikipedia says about the MDCT:

The modified discrete cosine transform (MDCT) is a lapped transform based on the type-IV discrete cosine transform (DCT-IV), with the additional property of being lapped: it is designed to be performed on consecutive blocks of a larger dataset, where subsequent blocks are overlapped so that the last half of one block coincides with the first half of the next block.

This overlapping, in addition to the energy-compaction qualities of the DCT, makes the MDCT especially attractive for signal compression applications, since it helps to avoid artifacts stemming from the block boundaries.

As a result of these advantages, the MDCT is employed in most modern lossy audio formats, including MP3, AC-3, Vorbis, Windows Media Audio, ATRAC, Cook, and AAC.

The implementation used here is based on the FFT wrapper published in [Borsi].

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. [Borsi] Marina Bosi, Richard E. Goldberg and Leonardo Chiariglione,
   "Introduction to Digital Audio Coding and Standards", Kluwer Academic Publishers, 01 December, 2002.
