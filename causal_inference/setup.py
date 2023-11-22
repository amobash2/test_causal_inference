from setuptools import setup, find_packages

setup(
    name="ci_pkg",
    version="0.0.0a0",
    description=("Causal Inference Functionalities"),
    python_requires='<3.11',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Causal Inference Functionalities",
        "Operating System :: Windows",
        "Operating System :: POSIX"],
)