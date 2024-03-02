import os
from distutils.core import setup, Extension
import numpy as np

libs = ["S4", "stdc++"]
lib_dirs = [os.path.join("src", "./build")]
libs.extend([lib[2::] for lib in "-L/home/bench/prg/mambaforge/envs/S4/lib/ -lblas -L/home/bench/prg/mambaforge/envs/S4/lib/ -llapack -L/home/bench/prg/mambaforge/envs/S4/lib/ -lfftw3 -L/home/bench/prg/mambaforge/envs/S4/lib/ -lpthread -I//home/bench/prg/mambaforge/envs/S4/lib -lcholmod -lamd -lcolamd -lcamd -lccolamd -L//home/bench/prg/mambaforge/envs/S4/lib/libmpi.so -L/home/bench/prg/mambaforge/envs/S4/lib/ -lboost_serialization".split()])
include_dirs = [np.get_include()]
libfile = os.path.join("src", "./build/libS4.a")
extra_link_args = [libfile]
sources = [os.path.join("src", "S4/main_python.c")]
S4module = Extension("S4ext",
	sources = sources,
	libraries = libs,
	library_dirs = lib_dirs,
    include_dirs = include_dirs,
    extra_objects = [libfile],
	extra_compile_args=["-std=gnu99"] 
)

setup(name = "S4",
	ext_modules = [S4module],
)
