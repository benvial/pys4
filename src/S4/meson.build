
# Link flags
lflags = ['-llapack', '-lblas', '-lfftw3', '-lpthread','-lcholmod', 
        '-lamd', '-lcolamd', '-lcamd', '-lccolamd', '-lmpi', '-lboost_serialization']

# Compiler-specific options
cflags = ['-Wall', '-O3', '-m64', '-march=native', '-mtune=native', 
        '-msse3', '-msse2', '-msse','-std=gnu99','-Wno-unused-parameter',
        '-Wno-unused-function','-Wno-unused-variable','-Wno-sign-compare']

cppflags = ['-Wall', '-DHAVE_BLAS','-DHAVE_LAPACK','-DHAVE_FFTW3', 
      '-DHAVE_LIBPTHREAD', '-DHAVE_LIBCHOLMOD', '-DHAVE_MPI', '-DHAVE_UNISTD_H',
      '-Wno-unused-parameter','-Wno-unused-function','-Wno-unused-variable',
      '-Wno-sign-compare']


boost =  dependency('boost_serialization', modules: ['serialization'], required:false)
if boost.found() == false
  # find boost installed with conda/mamba
  cpp_compiler = meson.get_compiler('cpp')
  boost_dep_tmp = cpp_compiler.find_library('boost_serialization')
  incdir_python= run_command(py,
      ['-c', 'import os; from sysconfig import get_paths; print(os.path.abspath(os.path.dirname(get_paths()["include"])))'],
      check : true
  ).stdout().strip()
  include_boost = incdir_python
  if run_command('[', '-d', include_boost, ']', check: true).returncode() != 0
    include_boost = '/usr/include'
  endif
  # include_boost = '/usr/include'
  boost = declare_dependency(dependencies : [ boost_dep_tmp ],
    include_directories: include_boost)
endif


blas = dependency(['openblas', 'OpenBLAS'])
lapack = dependency(['openblas', 'OpenBLAS'])
fftw = dependency(['fftw3', 'fftw'])
mpi = dependency('mpich',required: false)
numpy = dependency('numpy',required: true)

dependencies = [py_dep, boost, blas, lapack, fftw,numpy]
if mpi.found()
  dependencies += mpi
endif


subdir('S4')

