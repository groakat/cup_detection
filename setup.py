from __future__ import with_statement

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

## windows install part from http://matthew-brett.github.io/pydagogue/installing_scripts.html
import os
from os.path import join as pjoin, splitext, split as psplit
from distutils.core import setup
from distutils.command.install_scripts import install_scripts
from distutils import log
from setuptools import find_packages


setup(
    name = "cup-detector",
    version = "0.1.0",
    author = "Peter Rennert",
    author_email = "p.rennert@cs.ucl.ac.uk",
    description = ("Cup Detection for Experiment"),
    license = "--",
    keywords = "video",
    url = "https://github.com/groakat/cup-detection",
    packages=find_packages(),
    # long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ],
    package_data = {
        '': ['*.svg', '*.yaml', '*.zip', '*.ico', '*.bat']
    }
)