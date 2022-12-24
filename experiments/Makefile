# Makefile the datascope-experiments module.

# Summary and context path of this makefile.
define SUMMARY
This the Makefile of the datascope-experiments sub-module which is intended to support \
experiment specific development tasks.
endef
PYTHON := $(shell which python)

# Paths to the parent directory of this makefile and the repo root directory.
MY_DIR_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR_PATH := $(realpath $(MY_DIR_PATH)..)
VENV_DIR := .venv

# Include common make functions.
include $(ROOT_DIR_PATH)/dev/makefiles/show-help.mk

.PHONY: setup
## Install the base bundle of datascope to the current python environment.
setup: requirements.txt
	pip install -e .

.PHONY: version
## Update the version file to increment the patch version of the module.
version:
	$(ROOT_DIR_PATH)/dev/scripts/increment-version.py datascope/experiments/version.py --patch

.PHONY: package
## Package into .whl and source code archive (.tar.gz).
package:
	python -m build

.PHONY: publish-test
## Publish to pypi (test version at test.pypi.org using the easeml account)
publish-test:
	twine upload --username easeml --repository-url https://test.pypi.org/legacy/ dist/*.tar.gz

.PHONY: publish
## Publish to pypi (using the API token stored in the PYPI_API_TOKEN_DATASCOPE_EXPERIMENTS environment variable)
publish:
	$(eval TWINE_PASSWORD := $(PYPI_API_TOKEN_DATASCOPE_EXPERIMENTS))
	$(eval export TWINE_PASSWORD)
	twine upload --username __token__ dist/*.tar.gz

.PHONY: publish-clean
## remove the distribution files
publish-clean:
	rm -rf dist/
	rm -rf datascope.egg-info
