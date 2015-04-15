#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        # Name of the project
        name='mdct',

        # Version
        version='0.1',

        # Description
        description='A fast MDCT implementation using FFTW.',

        # Your contact information
        author='Nils Werner',
        author_email='nils.werner@gmail.com',

        # License
        license='MIT',

        # Packages in this project
        # find_packages() finds all these automatically for you
        packages=setuptools.find_packages(),

        # Dependencies, this installs the entire Python scientific
        # computations stack
        install_requires=[
            'numpy>=1.6',
            'scipy>=0.13.0',
            'stft',
        ],

        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        },
        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],

        classifiers=[
            'Development Status :: 1 - Planning',
            'Environment :: Console',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'License :: Other/Proprietary License',
            'Programming Language :: Python :: 2.7',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],
        zip_safe=True,
        include_package_data=True,
    )
