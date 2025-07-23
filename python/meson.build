project(
  'tskit', 'c',
  version: run_command(['tskit' / '_version.py'], check: true).stdout().strip()
)

subdir('tskit')
py = import('python').find_installation(pure: false)
py.install_sources(src, subdir: 'tskit')

deps = [subproject('libtskit').get_variable('tskit_dep')]
deps += declare_dependency(
  include_directories: run_command(
    py, '-c', 'import numpy;print(numpy.get_include())',
    check: true
  ).stdout().strip()
)

py.extension_module(
  '_tskit',
  '_tskitmodule.c',
  dependencies: deps,
  include_directories: 'lwt_interface',
  install: true,
)
