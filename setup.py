#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='mdct',
        version='0.3',

        description='A fast MDCT implementation using SciPy and FFTs',
        author='Nils Werner',
        author_email='nils.werner@gmail.com',

        license='MIT',
        packages=setuptools.find_packages(),

        install_requires=[
            'numpy>=1.6',
            'scipy>=0.13.0',
            'stft>=0.5',
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
            'Development Status :: 4 - Beta',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=True,
        include_package_data=True,
    )
