#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.core', 'numpy']))
from bob.blitz.extension import Extension
from bob.extension.utils import uniq
import bob.core

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'bob', 'math', 'include')
include_dirs = [package_dir, bob.core.get_include()]

# uses LAPACK/BLAS info from numpy
import numpy.__config__ as npconf

lapack_library_dirs = []
lapack_libraries = []
lapack_include_dirs = []
lapack_define_macros = []
if hasattr(npconf, 'lapack_info') and npconf.lapack_info:
  lapack_library_dirs = npconf.lapack_info['library_dirs']
  lapack_libraries = npconf.lapack_info['libraries']
  lapack_include_dirs = npconf.lapack_info['include_dirs']
  lapack_define_macros = npconf.lapack_info['define_macros']
if hasattr(npconf, 'lapack_opt_info') and npconf.lapack_opt_info:
  lapack_library_dirs = npconf.lapack_opt_info['library_dirs']
  lapack_libraries = npconf.lapack_opt_info['libraries']
  lapack_include_dirs = npconf.lapack_opt_info['include_dirs']
  lapack_define_macros = npconf.lapack_opt_info['define_macros']
if hasattr(npconf, 'lapack_mk_info') and nnpconf.lapack_mkl_info:
  lapack_library_dirs = npconf.lapack_mkl_info['library_dirs']
  lapack_libraries = npconf.lapack_mkl_info['libraries']
  lapack_include_dirs = npconf.lapack_mkl_info['include_dirs']
  lapack_define_macros = npconf.lapack_mkl_info['define_macros']

blas_library_dirs = []
blas_libraries = []
blas_include_dirs = []
blas_define_macros = []
if hasattr(npconf, 'blas_info') and npconf.blas_info:
  blas_library_dirs = npconf.blas_info['library_dirs']
  blas_libraries = npconf.blas_info['libraries']
  blas_include_dirs = npconf.blas_info['include_dirs']
  blas_define_macros = npconf.blas_info['define_macros']
if hasattr(npconf, 'blas_opt_info') and npconf.blas_opt_info:
  blas_library_dirs = npconf.blas_opt_info['library_dirs']
  blas_libraries = npconf.blas_opt_info['libraries']
  blas_include_dirs = npconf.blas_opt_info['include_dirs']
  blas_define_macros = npconf.blas_opt_info['define_macros']
if hasattr(npconf, 'blas_mk_info') and nnpconf.blas_mkl_info:
  blas_library_dirs = npconf.blas_mkl_info['library_dirs']
  blas_libraries = npconf.blas_mkl_info['libraries']
  blas_include_dirs = npconf.blas_mkl_info['include_dirs']
  blas_define_macros = npconf.blas_mkl_info['define_macros']

# mix-in
library_dirs = uniq(lapack_library_dirs + blas_library_dirs)
libraries = uniq(lapack_libraries + blas_libraries)
include_dirs = uniq(include_dirs + lapack_include_dirs + blas_include_dirs)
define_macros = uniq(lapack_define_macros + blas_define_macros)

version = '2.0.0a0'

setup(

    name='bob.math',
    version=version,
    description='Bindings for bob.math',
    url='http://github.com/bioidiap/bob.math',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.extension',
    ],

    namespace_packages=[
      "bob",
      ],

    ext_modules = [
      Extension("bob.math.version",
        [
          "bob/math/version.cpp",
          ],
        version = version,
        include_dirs = include_dirs,
        ),
      Extension("bob.math._library",
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

          "bob/math/histogram.cpp",
          "bob/math/linsolve.cpp",
          "bob/math/pavx.cpp",
          "bob/math/norminv.cpp",
          "bob/math/scatter.cpp",
          "bob/math/lp_interior_point.cpp",
          "bob/math/main.cpp",
          ],
        version = version,
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        libraries = libraries,
        define_macros = define_macros,
      ),
    ],

    entry_points={
      },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
