#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
from contextlib import suppress
from distutils.core import Command, Extension, setup
from pathlib import Path

import numpy as np
from setuptools.command.build import build

lib_dirs = ["src/S4/build"]
libs = ["S4", "stdc++"]
libs.extend(
    [
        "blas",
        "lapack",
        "fftw3",
        "pthread",
        "cholmod",
        "amd",
        "colamd",
        "camd",
        "ccolamd",
        "boost_serialization",
    ]
)
include_dirs = [np.get_include()]
libfile = "src/S4/build/libS4.a"
sources = ["src/S4/S4/main_python.c"]


# print(include_dirs)
# print(lib_dirs)
# print(libs)

S4module = Extension(
    "S4",
    sources=sources,
    libraries=libs,
    library_dirs=lib_dirs,
    include_dirs=include_dirs,
    extra_objects=[libfile],
    extra_compile_args=["-std=gnu99"],
)


class CustomCommand(Command):
    def initialize_options(self):
        self.bdist_dir = None

    def finalize_options(self):
        with suppress(Exception):
            self.bdist_dir = Path(self.get_finalized_command("bdist_wheel").bdist_dir)

    def run(self):
        subprocess.run(
            ["make", "lib"],
            capture_output=True,
            check=True,
        )


class CustomBuild(build):
    sub_commands = [("build_custom", None)] + build.sub_commands


setup(
    name="pys4",
    ext_modules=[S4module],
    packages=["pys4"],
    cmdclass={"build": CustomBuild, "build_custom": CustomCommand},
)
