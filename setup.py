import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


def get_long_description():
    return open("README.md", "r").read()


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f:
            requirements.append(line.strip())
    return requirements


setup(
    name="shoeprint-recognition",
    version="1.0.4",
    keywords=("python", "tensorflow", "deeplearning", "shoeprint"),
    description="“恒锐杯”鞋印花纹图像类别判定挑战赛",
    long_description=get_long_description(),
    license="MIT",

    url="https://zsync.github.io/shoeprint-recognition/",
    author="SigureMo",
    author_email="sigure_mo@163.com",

    platforms="any",
    install_requires=get_requirements(),

    scripts=[],
    ext_modules=cythonize(Extension(
        'deformation',
        sources=['data_loader/deformation.pyx'],
        language='c',
        include_dirs=[np.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[],
        extra_link_args=[]
    )),
    cmdclass={'build_ext': build_ext}
)
