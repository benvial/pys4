
# Link flags
lflags = ['-llapack', '-lblas', '-lfftw3', '-lpthread', '-lcholmod', 
        '-lamd', '-lcolamd', '-lcamd', '-lccolamd', '-lmpi',
        '-lboost_serialization', '-lfabric']

# Compiler-specific options
cflags = ['-Wall', '-O3', '-m64', '-march=native', '-mtune=native', 
        '-msse3', '-msse2', '-msse', '-Wno-unused-parameter',
        '-Wno-unused-function', '-Wno-unused-variable', '-Wno-sign-compare',
        '-Wno-maybe-uninitialized', '-Wno-unused-but-set-variable',
        '-Wno-incompatible-pointer-types', '-Wno-missing-field-initializers', 
        '-Wno-cast-function-type']

cppflags = ['-Wall', '-DHAVE_BLAS', '-DHAVE_LAPACK', '-DHAVE_FFTW3', 
      '-DHAVE_LIBPTHREAD', '-DHAVE_LIBCHOLMOD', '-DHAVE_MPI', '-DHAVE_UNISTD_H',
      '-Wno-unused-parameter', '-Wno-unused-function', '-Wno-unused-variable',
      '-Wno-sign-compare', '-Wno-class-memaccess', '-Wno-maybe-uninitialized',
      '-Wno-narrowing', '-Wno-unused-but-set-variable']

conda = get_option('conda')
boost =  dependency('boost_serialization', modules: ['serialization'], required:false)
if boost.found() == false
  # find boost installed with conda/mamba
  cpp_compiler = meson.get_compiler('cpp')
  boost_dep_tmp = cpp_compiler.find_library('boost_serialization')
  if conda
    incdir_python= run_command(py,
        ['-c', 'import os; from sysconfig import get_paths; print(os.path.abspath(os.path.dirname(get_paths()["include"])))'],
        check : true
    ).stdout().strip()
    include_boost = incdir_python
  else
    include_boost = '/usr/include'
  endif
  # include_boost = '/usr/include'
  boost = declare_dependency(dependencies : [ boost_dep_tmp ],
    include_directories: include_boost)
endif

ss_dep =  dependency('suitesparse', required:false)
if ss_dep.found() == false
  # find suitesparse installed with conda/mamba
  c_compiler = meson.get_compiler('c')
  umfpack_dep_tmp = c_compiler.find_library('umfpack')
  if conda
    incdir_python= run_command(py,
        ['-c', 'import os; from sysconfig import get_paths; print(os.path.dirname(get_paths()["include"]))'],
        check : true
    ).stdout().strip()
    inc_ss = incdir_python + '/suitesparse'
  else
    inc_ss = '/usr/include/suitesparse'
  endif
  ss_dep = declare_dependency(dependencies : [ umfpack_dep_tmp ],
    include_directories: inc_ss)
endif

blas = dependency(['openblas', 'OpenBLAS'])
lapack = dependency(['openblas', 'OpenBLAS'])
fftw = dependency(['fftw3', 'fftw'])
mpi = dependency('mpich',required: false)
numpy = dependency('numpy',required: true)

dependencies = [py_dep, boost, blas, lapack, fftw, ss_dep, numpy]
if mpi.found()
  dependencies += mpi
endif


subdir('S4')

