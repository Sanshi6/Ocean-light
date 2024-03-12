# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
#
# setup(
#     ext_modules=cythonize([Extension("region", ["region.pyx", "src/region.c"])]),
# )
#

from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

ext_modules = [
    Extension("region", ["region.pyx", "src/region.c"], include_dirs=['.'],)
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3})
)