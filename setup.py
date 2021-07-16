import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages


setup(
    name="benptpy",
    version="2.0.1",
    description="Bindings to a CUDA path tracer",
    author="Benjamin Chislett",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    cmake_install_dir="python/benptpy",
    include_package_data=True,
)
