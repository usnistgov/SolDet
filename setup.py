#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:56:15 2022

@author: sjguo
"""
from setuptools import setup

setup(
    name='soldet',
    version='0.0.1',
    packages=['soldet'],
    long_description=open('README.md').read(),
    install_requires = [
        'numpy == 1.22.0',
        'scikit-learn==0.23.1',
        'scipy==1.6.3',
        'tensorflow==2.7.2',
        'tqdm==4.47.0',
        'matplotlib==3.2.2',
        'lmfit==1.0.1',
        'h5py==3.1.0',
        'pandas==1.0.5',
        'seaborn==0.10.1']
)

