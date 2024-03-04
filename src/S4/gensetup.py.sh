#!/bin/bash

OBJDIR="$1"
LIBFILE="$2"
LIBS="$3"

echo "LIBFILE: $LIBFILE"

cat <<SETUPPY > setup.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from distutils.core import setup, Extension
import numpy as np

libs = ["S4", "stdc++"]
lib_dirs = [os.path.join("src", "$OBJDIR")]
libs.extend([lib[2::] for lib in "$LIBS".split()])
include_dirs = [np.get_include()]
libfile = os.path.join("src", "$LIBFILE")
extra_link_args = [libfile]
sources = [os.path.join("src", "S4/main_python.c")]
S4module = Extension("pys4._S4ext",
	sources = sources,
	libraries = libs,
	library_dirs = lib_dirs,
    include_dirs = include_dirs,
    extra_objects = [libfile],
	extra_compile_args=["-std=gnu99"] 
)

setup(name = "pys4",
	ext_modules = [S4module],
)
SETUPPY
