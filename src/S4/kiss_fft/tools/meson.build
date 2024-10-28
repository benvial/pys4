
lib = static_library('kiss_fftnd','kiss_fftnd.c', name_suffix: 'o',
  dependencies: dependencies, link_args: lflags, c_args: cflags, cpp_args: cppflags, 
    include_directories: include_directories, pic: true, install: false)

S4_libraries += lib