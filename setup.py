#!/usr/bin/env python

import setuptools 

with open("README.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name="differentiable_filters", 
    version="0.0.1",
    author='Alina Kloss, MPI-IS Tuebingen, Autonomous Motion',
    author_email='alina.kloss@yahoo.de',
    description="TensorFlow code for differentiable filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/akloss/differentiable_filters',
    packages=setuptools.find_packages(),
    install_requires=['tensorflow-gpu==1.14.0', 'matplotlib', 
                      'numpy==1.16.4', 'tensorflow-probability==0.7.0',
                      'gast==0.2.2', 'pyaml'],
    python_requires='>=3.6',
)

