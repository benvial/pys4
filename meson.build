


project('S4','cpp', 'c',
  meson_version: '>=1.1',
  default_options : ['warning_level=2'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

subdir('pys4')
subdir('src')