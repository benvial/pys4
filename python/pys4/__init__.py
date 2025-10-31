#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: The pyS4 Developers
# This file is part of pyS4
# License: GPLv3

"""
The pyS4 package for simulating electromagnetic waves in periodic structures.
"""

__all__ = ["Simulation", "SpectrumSampler"]

from . import _S4
from .__about__ import __author__, __description__, __version__
from ._core import *

_S4.__version__ = "1.1"
_S4.__description__ = _S4.__doc__
_S4.__author__ = "Victor Liu"
_S4.__email__ = "victorliu@alumni.stanford.edu"
