# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Copyright (c) 2022 MeteoSwiss, Bruno Zuercher.
# Published under the BSD-3-Clause license.
#------------------------------------------------------------------------------

from setuptools import setup

setup(
    name='fast-barnes-py',
    version='1.0',
    description='Fast Barnes Interpolation',
    author='Bruno Zürcher',
    author_email='bruno.zuercher@meteoswiss.ch',
    url='https://github.com/MeteoSwiss/fast-barnes-py/',
    license='BSD 3-Clause License',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['fastbarnes', 'fastbarnes.util'],
    install_requires=['numpy', 'scipy', 'numba', 'matplotlib', 'basemap', 'Pillow'],
    python_requires='>=3.7',
    keywords=['python', 'interpolation', 'gridding'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science / Research',
        'License :: OSI Approved :: BSD License',
        'Operating System:: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific / Engineering ',
        'Topic :: Mathematics',
        'Topic :: Visualization',
    ],
)
