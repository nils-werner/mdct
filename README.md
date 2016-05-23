MDCT
====

[![Build Status](https://travis-ci.org/audiolabs/mdct.svg?branch=master)](https://travis-ci.org/audiolabs/mdct)
[![Docs Status](https://readthedocs.org/projects/mdct/badge/?version=latest)](https://mdct.readthedocs.org/en/latest/)

A fast MDCT implementation using SciPy and FFTs


Installation
------------

As usual

    pip install mdct


## Dependencies

 - NumPy
 - SciPy
 - STFT


Usage
-----


    import mdct
    
    spectrum = mdct.mdct(sig)


**Also see the [docs](http://mdct.readthedocs.io/)**

References
----------

 - Implementation: Marina Bosi, Richard E. Goldberg and Leonardo Chiariglione, "Introduction to Digital Audio Coding and Standards", Kluwer Academic Publishers, 01 December, 2002.
