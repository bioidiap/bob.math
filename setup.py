#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz','xbob.extension']))
from xbob.blitz.extension import Extension
import xbob.extension

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'xbob', 'math', 'include')
include_dirs = [package_dir]

packages = ['bob-math >= 1.2.2']
version = '2.0.0a0'

setup(

    name='xbob.math',
    version=version,
    description='Bindings for bob.math',
    url='http://github.com/bioidiap/xbob.math',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
      'xbob.extension',
    ],

    namespace_packages=[
      "xbob",
      ],

    ext_modules = [
      Extension("xbob.math.version",
        [
          "xbob/math/version.cpp",
          ],
        packages = packages,
        version = version,
        include_dirs = include_dirs,
        ),
      Extension("xbob.math._library",
        [
          "xbob/math/histogram.cpp",
          "xbob/math/linsolve.cpp",
          "xbob/math/pavx.cpp",
          "xbob/math/norminv.cpp",
          "xbob/math/scatter.cpp",
          "xbob/math/lp_interior_point.cpp",
          "xbob/math/main.cpp",
          ],
        packages = packages,
        version = version,
        include_dirs = include_dirs,
#        define_macros = [('XBOB_SHORT_DOCSTRINGS',1)],
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
