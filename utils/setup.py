from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
setup(
    name = "functions_cython",
    ext_modules = cythonize('cython_func.pyx'),  # accepts a glob pattern
    include_dirs = [np.get_include()]
)
setup(
    ext_modules =
    [
        Extension("cython_func", ["cython_func.c"],
                  include_dirs = [np.get_include()])
    ],
)


