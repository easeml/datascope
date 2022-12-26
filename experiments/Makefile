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

.PHONY: lint
## Run linter to automatically test for formatting errors.
lint:
	flake8 --max-line-length=120 --show-source --statistics -v datascope setup.py

.PHONY: format
## Run the formatter to automatically fix formatting errors.
format:
	black --line-length 120 -v datascope setup.py

.PHONY: version
## Update the version file to increment the patch version of the module.
version:
	$(ROOT_DIR_PATH)/dev/scripts/increment-version.py datascope/experiments/version.py --patch

.PHONY: package
## Package into a source distribution (sdist).
package:
	python -m build --no-isolation --sdist

.PHONY: publish-test
## Publish to pypi (using the API token stored in the PYPI_TEST_API_TOKEN_DATASCOPE_EXPERIMENTS environment variable).
publish-test:
	$(eval TWINE_PASSWORD := $(PYPI_TEST_API_TOKEN_DATASCOPE_EXPERIMENTS))
	$(eval export TWINE_PASSWORD)
	twine upload --username __token__ --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
## Publish to pypi (using the API token stored in the PYPI_API_TOKEN_DATASCOPE_EXPERIMENTS environment variable)
publish:
	$(eval TWINE_PASSWORD := $(PYPI_API_TOKEN_DATASCOPE_EXPERIMENTS))
	$(eval export TWINE_PASSWORD)
	twine upload --username __token__ dist/*

.PHONY: publish-clean
## remove the distribution files
publish-clean:
	rm -rf dist/
	rm -rf datascope.egg-info
