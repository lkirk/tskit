project(
  'tskit', 'c',
  version: run_command(['tskit' / '_version.py'], check: true).stdout().strip(),
  default_options: ['c_std=c99', 'cpp_std=c++11']
)

subdir('tskit')
py = import('python').find_installation(pure: false)
py.install_sources(src, subdir: 'tskit')

deps = [
  subproject(
    'libtskit', default_options: ['werror=false']
  ).get_variable('tskit_dep'),
  declare_dependency(
    include_directories: run_command(
      py, '-c', 'import numpy;print(numpy.get_include())',
      check: true
    ).stdout().strip()
  )
]

py.extension_module(
  '_tskit',
  '_tskitmodule.c',
  dependencies: deps,
  include_directories: 'lwt_interface',
  install: true,
)
