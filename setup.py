#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz','bob.extension']))
from bob.blitz.extension import Extension
import bob.extension

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'bob', 'math', 'include')
include_dirs = [package_dir]

packages = ['bob-math >= 1.2.2']
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
        packages = packages,
        version = version,
        include_dirs = include_dirs,
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
        packages = packages,
        version = version,
        include_dirs = include_dirs,
#        define_macros = [('BOB_SHORT_DOCSTRINGS',1)],
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
