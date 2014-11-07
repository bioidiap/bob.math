#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz', 'numpy'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext
from bob.extension.utils import uniq, find_library

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

import os

def get_flags(keys):
  """Returns link/include flags for LAPACK/BLAS based on what NumPy uses

  In case NumPy is using a fallback (i.e., no LAPACK/BLAS installed on the host
  system), then defaults to linking against 'lapack' and 'blas' and hope it
  works.
  """

  import numpy.__config__ as npconf

  retval = dict(
      library_dirs = [],
      libraries = [],
      system_include_dirs = [],
      define_macros = [],
      extra_compile_args = [],
      extra_link_args = [],
      )

  for key in keys:

    if not hasattr(npconf, key): continue
    obj = getattr(npconf, key)
    if not obj: continue #it is empty

    retval = dict(
        library_dirs = obj.get('library_dirs', []),
        libraries = obj.get('libraries', []),
        system_include_dirs = obj.get('include_dirs', []),
        define_macros = obj.get('define_macros', []),
        extra_compile_args = obj.get('extra_compile_args', []),
        extra_link_args = obj.get('extra_link_args', []),
        )

  return retval

lapack_flags = get_flags(['lapack_info', 'lapack_opt_info', 'lapack_mkl_info'])
blas_flags = get_flags(['blas_info', 'blas_opt_info', 'blas_mkl_info'])

# mix-in
math_flags = dict(
    library_dirs = [],
    libraries = [],
    system_include_dirs = [],
    define_macros = [],
    extra_compile_args = [],
    extra_link_args = [],
    )
for key in math_flags:
  math_flags[key] = uniq(lapack_flags.get(key, []) + blas_flags.get(key, []))

# checks if any libraries are being linked, otherwise we
# search through the filesystem in stock locations.
if not math_flags['libraries']:
  # reset all entries
  math_flags = dict(
      library_dirs = [],
      libraries = [],
      system_include_dirs = [],
      define_macros = [],
      extra_compile_args = [],
      extra_link_args = [],
      )

  # tries first to find an MKL implementation
  lapack = find_library('mkl_lapack64')
  if not lapack:
    # if that fails, go for the default implementation
    lapack = find_library('lapack', subpaths=['sse2', ''])

  if not lapack:
    print("ERROR: LAPACK library not found - have that installed or set BOB_PREFIX_PATH to point to the correct installation prefix")
    sys.exit(1)

  # tries first to find an MKL implementation
  blas = find_library('mkl')
  if not blas:
    # if that fails, go for the default implementation of cblas
    blas = find_library('cblas', subpaths=['sse2', ''])
  if not blas:
    # if that fails, go for the default implementation of blas
    blas = find_library('blas', subpaths=['sse2', ''])

  if not blas:
    print("ERROR: BLAS library not found - have that installed or set BOB_PREFIX_PATH to point to the correct installation prefix")
    sys.exit(1)

  # at this point both lapack and blas were detected, proceed
  def libname(f): return os.path.splitext(os.path.basename(f))[0][3:]

  math_flags['library_dirs'] = uniq([
    os.path.dirname(lapack[0]),
    os.path.dirname(blas[0]),
    ])
  math_flags['libraries'] = uniq([
    libname(lapack[0]),
    libname(blas[0])
    ])

  print("\nLAPACK/BLAS configuration from filesystem scan:")

else:

  print("\nLAPACK/BLAS configuration from NumPy:")

print(" * system include directories: %s" % ', '.join(math_flags['system_include_dirs']))
print(" * defines: %s" % \
  ', '.join(['-D%s=%s' % k for k in math_flags['define_macros']]))
print(" * linking arguments: %s" % ', '.join(math_flags['extra_link_args']))
print(" * libraries: %s" % ', '.join(math_flags['libraries']))
print(" * library directories: %s\n" % ', '.join(math_flags['library_dirs']))

setup(

    name='bob.math',
    version=version,
    description='Mathematical functions of Bob',
    url='http://github.com/bioidiap/bob.math',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    setup_requires = build_requires,
    install_requires = build_requires,

    namespace_packages=[
      "bob",
    ],

    ext_modules = [
      Extension("bob.math.version",
        [
          "bob/math/version.cpp",
        ],
        version = version,
        system_include_dirs = math_flags.get('system_include_dirs', []),
        library_dirs = math_flags.get('library_dirs', []),
        libraries = math_flags.get('libraries', []),
        define_macros = math_flags.get('define_macros', []),
        extra_compile_args = math_flags['extra_compile_args'],
        extra_link_args = math_flags.get('extra_link_args', []),
        bob_packages = bob_packages,
      ),

      Library("bob.math.bob_math",
        [
          "bob/math/cpp/det.cpp",
          "bob/math/cpp/eig.cpp",
          "bob/math/cpp/inv.cpp",
          "bob/math/cpp/linsolve.cpp",
          "bob/math/cpp/log.cpp",
          "bob/math/cpp/LPInteriorPoint.cpp",
          "bob/math/cpp/lu.cpp",
          "bob/math/cpp/norminv.cpp",
          "bob/math/cpp/pavx.cpp",
          "bob/math/cpp/pinv.cpp",
          "bob/math/cpp/svd.cpp",
          "bob/math/cpp/sqrtm.cpp",
        ],
        version = version,
        bob_packages = bob_packages,
        system_include_dirs = math_flags['system_include_dirs'],
        library_dirs = math_flags['library_dirs'],
        libraries = math_flags['libraries'],
        define_macros = math_flags['define_macros'],
      ),

      Extension("bob.math._library",
        [
          "bob/math/histogram.cpp",
          "bob/math/linsolve.cpp",
          "bob/math/pavx.cpp",
          "bob/math/norminv.cpp",
          "bob/math/scatter.cpp",
          "bob/math/lp_interior_point.cpp",
          "bob/math/main.cpp",
        ],
        version = version,
        bob_packages = bob_packages,
        system_include_dirs = math_flags['system_include_dirs'],
        library_dirs = math_flags['library_dirs'],
        libraries = math_flags['libraries'],
        define_macros = math_flags['define_macros'],
        extra_compile_args = math_flags['extra_compile_args'],
        extra_link_args = math_flags['extra_link_args'],
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
