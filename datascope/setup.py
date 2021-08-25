from distutils.core import setup

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")

setup(
    name="datascope",
    version="0.1",
    packages=["datascope",],
    license="MIT",
    description="Measuring data importance over ML pipelines using the Shapley value",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=reqs,
)
