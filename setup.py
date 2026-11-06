import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "simcore.fastio.ring_buffer",
        ["src/simcore/fastio/ring_buffer.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        "simcore.fastio.trajectory_reader",
        ["src/simcore/fastio/trajectory_reader.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
