
c_sources = ['pattern','intersection','predicates']
foreach libname : c_sources

  lib = static_library(libname,libname + '.c', name_suffix: 'o',
    dependencies: dependencies, link_args: lflags, c_args: cflags, cpp_args: cppflags, 
      include_directories: include_directories, pic: true, install: false)
  S4_libraries += lib
endforeach
