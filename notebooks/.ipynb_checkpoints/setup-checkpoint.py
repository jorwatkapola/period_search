from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("wwzs.pyx"),
#    package_dir={'cython_test': '/data/jkok1g14/ogle_xrom/period_search/notebooks'},
    include_dirs=[numpy.get_include()]
)