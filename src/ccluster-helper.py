from setuptools import setup
from Cython.Build import cythonize

import numpy
# cimport numpy
# python ccluster-helper.py build_ext --inplace
setup(
    ext_modules = cythonize("ccluster.pyx"),
    include_dirs=[numpy.get_include()]
)