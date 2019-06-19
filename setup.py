#! /usr/bin/env python
# encoding: utf-8

from __future__ import unicode_literals

from setuptools import find_packages, setup

extras_require = {
    "test": [
        "mock>=2.0.0,<3.0",
        "pytest>=4.3.1,<5.0",
    ]
}

setup(
    name="automatic-data-generation",
    version="0.1.0",
    description="Text generation library",
    license=None,
    author="The Snips Team",
    packages=find_packages(),
    install_requires=[
        "snips-nlu",
        "snips-nlu-metrics",
        "torch",
        "torchvision",
        "torchtext",
        "tensorboardX==1.4",
        "tb-nightly",
        "nltk",
        "spacy",
        "pandas"
    ],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
)
