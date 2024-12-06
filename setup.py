# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

from setuptools import setup

setup(
    name='fast-barnes-py',
    version='2.0.0',
    description='Fast Barnes Interpolation',
    author='Bruno Zürcher',
    author_email='bruno.zuercher@meteoswiss.ch',
    url='https://github.com/MeteoSwiss/fast-barnes-py/',
    license='BSD 3-Clause License',
    long_description=open('README.md').read(),
#    long_description=open('doc/PyPI_Doc.md').read(),
    long_description_content_type='text/markdown',
    packages=['fastbarnes', 'fastbarnes.util'],
    install_requires=['numpy', 'scipy', 'numba'],
    python_requires='>=3.9, <3.13',
    keywords=['python', 'interpolation', 'gridding', 'inverse distance weighting', 'fast algorithms', 'Gaussian', 'convolution'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Oceanography',
    ],
)
