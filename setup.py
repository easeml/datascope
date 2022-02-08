from distutils.core import setup


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_requires = parse_requirements("requirements.txt")
extras_require = {"dev": parse_requirements("requirements-dev.txt"), "exp": parse_requirements("requirements-exp.txt")}
extras_require_all = set(r for e in extras_require.values() for r in e)
extras_require["complete"] = extras_require_all

setup(
    name="datascope",
    version="0.1",
    packages=["datascope", "datascope.algorithms", "datascope.utils", "datascope.inspection", "datascope.importance"],
    license="MIT",
    description="Measuring data importance over ML pipelines using the Shapley value.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require,
)
