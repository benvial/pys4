
include_directories = include_directories(['.', 'RNP', 'kiss_fft'])

S4_libraries = []

cpp_sources = ['S4', 'rcwa']

foreach libname : cpp_sources
  lib = static_library(libname,libname + '.cpp', name_suffix: 'o',
    dependencies: dependencies, link_args: lflags, c_args: cflags, cpp_args: cppflags, 
      include_directories: include_directories, pic: true, install: false)
    S4_libraries += lib
endforeach


c_sources = ['SpectrumSampler','cubature','convert', 'sort', 'numalloc', 'gsel']

foreach libname : c_sources
  lib = static_library(libname,libname + '.c', name_suffix: 'o',
    dependencies: dependencies, link_args: lflags, c_args: cflags, cpp_args: cppflags, 
      include_directories: include_directories, pic: true, install: false)
    S4_libraries += lib
endforeach


subdir('fmm')
subdir('pattern')
subdir('kiss_fft')
subdir('RNP')

py.extension_module('_S4',
  'main_python.c',
  include_directories: include_directories,
  dependencies : dependencies,
  link_args: lflags,
  c_args: cflags, cpp_args: cppflags, 
  install : true,
  link_whole: S4_libraries,
  subdir: 'pys4'
)
