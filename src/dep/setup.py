from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('to_be_deleted_c.pyx'))