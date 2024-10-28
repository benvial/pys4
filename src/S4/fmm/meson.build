
cpp_sources = ['fft_iface', 'fmm_closed', 'fmm_common', 'fmm_experimental', 
      'fmm_FFT', 'fmm_kottke', 'fmm_PolBasisJones', 'fmm_PolBasisNV', 
      'fmm_PolBasisVL']

foreach libname : cpp_sources

  lib = static_library(libname,libname + '.cpp', name_suffix: 'o',
    dependencies: dependencies, link_args: lflags, c_args: cflags, cpp_args: cppflags, 
      include_directories: include_directories, pic: true, install: false)
  S4_libraries += lib
endforeach
