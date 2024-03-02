#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3

import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext


class Build(build_ext):
    """Customized setuptools build command."""

    def run(self):
        os.system("make install-S4")
        super().run()


setup(
    has_ext_modules=lambda: True,
    cmdclass={
        "build_ext": Build,
    },
)
