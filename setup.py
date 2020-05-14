#!/usr/bin/env python

from setuptools import setup

setup(name='imtools',
      version='0.1',
      description='A collection of image-processing utilities',
      url='',
      author='Mateusz Kozak',
      author_email='mateusz.kozak@unige.ch',
      license='MIT',
      packages=['imtools'],
      install_requires=[
          'numpy',
          'scikit-image'
      ],
      zip_safe=False)

