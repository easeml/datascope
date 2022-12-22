from setuptools import setup
from distutils.util import convert_path
from typing import Dict, Any


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


main_ns: Dict[str, Any] = {}
ver_path = convert_path("experiments/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

install_requires = parse_requirements("requirements.txt")

setup(
    name="datascope-experiments",
    version=main_ns["__version__"],
    packages=[
        "experiments",
        "experiments.datasets",
        "experiments.pipelines",
        "experiments.reports",
        "experiments.scenarios",
    ],
    entry_points={"console_scripts": ["datascope-experiments=experiments:main"]},
    license="MIT",
    author_email="easeml@ds3lab.com",
    url="https://ease.ml/datascope/",
    description="Module for running experiments on top of datascope.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
)
