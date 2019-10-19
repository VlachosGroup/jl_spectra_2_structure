from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "jl_structure_2_spectra: A method for deconvoluting spectra"
# Long description will go up on the pypi page
long_description = """

jl_structure_2_spectra
======================
This module trains on known experimental pure-component spectra
in order to deconvolute unknown mixture spectra.


License
=======
``jl_structure_2_spectra`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019--, Joshua Lansford, The University of Delaware.
"""

NAME = "jl_structure_2_spectra"
MAINTAINER = "Joshua Lansford"
MAINTAINER_EMAIL = "lansford.jl@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/JLANS/jl_structure_2_spectra"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Joshua Lansford"
AUTHOR_EMAIL = "lansford.jl@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'jl_structure_2_spectra': [pjoin('data', '*'),pjoin('data', '*','*')]}
REQUIRES = ["numpy","scipy","matplotlib"]
PYTHON_REQUIRES = ">= 3.5"
