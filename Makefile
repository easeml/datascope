# Makefile the datascope project.

# Summary and context path of this makefile.
SUMMARY := This the main Makefile of the datascope project intended to aid in all basic development tasks.
PYTHON := $(shell which python)

# Paths to the parent directory of this makefile and the repo root directory.
MY_DIR_PATH := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR_PATH := $(realpath $(MY_DIR_PATH))
VENV_DIR := .venv

# Include common make functions.
include $(ROOT_DIR_PATH)/dev/makefiles/show-help.mk

$(VENV_DIR)/touchfile:
	test -d $(VENV_DIR) || virtualenv $(VENV_DIR) --prompt "(datascope)" --python $(PYTHON)
	touch $(VENV_DIR)/touchfile

.PHONY: shell
## Load an instance of the shell with the appropriate virtual environment.
shell: $(VENV_DIR)/touchfile
#exec "bash -i <<< 'source $(VENV_DIR)/bin/activate; exec </dev/tty'"
# echo $(SHELL)
# exec "bash --init-file $(VENV_DIR)/bin/activate"
	. $(VENV_DIR)/bin/activate && exec bash

.PHONY: build
## Build the module (including all C extensions).
build:
	python setup.py build_ext --inplace

.PHONY: cython-build
## Build the module (including all C extensions) using cython.
cython-build:
	python setup.py build_ext --inplace --use-cython

.PHONY: setup
## Install the base bundle of datascope to the current python environment.
setup: requirements.txt
	pip install -e .

.PHONY: setup-dev
## Install the base and development bundles of datascope to the current python environment.
setup-dev: requirements-dev.txt
	pip install -e .[dev]

.PHONY: setup-exp
## Install the base and experimental bundles of datascope to the current python environment.
setup-exp: experiments/requirements.txt
	pip install -e experiments

.PHONY: setup-all
## Install all bundles of datascope to the current python environment.
setup-all: requirements.txt requirements-dev.txt experiments/requirements.txt
	pip install -e .[complete]
	pip install -e experiments

.PHONY: test
## Run tests (excluding benchmark tests).
test:
	pytest -k "not benchmark"

.PHONY: test-coverage
## Run tests (excluding benchmark tests) and produce a coverage report.
test-coverage:
	pytest -k "not benchmark" --cov=datascope --cov-report=xml:var/coverage.xml

.PHONY: test-benchmark
## Run benchmark tests.
test-benchmark:
	pytest -k "benchmark"

.PHONY: lint
## Run linter to automatically test for formatting errors.
lint:
	flake8 --max-line-length=120 --show-source --statistics -v datascope setup.py

.PHONY: format
## Run the formatter to automatically fix formatting errors.
format:
	black --line-length 120 -v datascope setup.py

.PHONY: clean-shell
## Clean the virtual environment and all the pyc files.
clean-shell:
	rm -rf $(VENV_DIR)
	find -iname "*.pyc" -delete

.PHONY: clean-package
## Clean all the package distribution directories.
clean-package:
	rm -rf dist/
	rm -rf datascope.egg-info

.PHONY: version
## Update the version file to increment the patch version of the module.
version:
	$(ROOT_DIR_PATH)/dev/scripts/update-version.py datascope/version.py --patch

.PHONY: version-minor
## Update the version file to increment the minor version of the module.
version-minor:
	$(ROOT_DIR_PATH)/dev/scripts/update-version.py datascope/version.py --minor

.PHONY: version-major
## Update the version file to increment the major version of the module.
version-major:
	$(ROOT_DIR_PATH)/dev/scripts/update-version.py datascope/version.py --major


.PHONY: tag
## Create a new git tag based on the current version.
tag:
	@$(eval VERSION=$(shell $(ROOT_DIR_PATH)/dev/scripts/update-version.py datascope/version.py))
	@echo VERSION=$(VERSION)
	git tag -a v$(VERSION) -m "Release v$(VERSION)"

.PHONY: package
## Package into a source distribution (sdist).
package:
	python -m build --no-isolation --sdist

.PHONY: publish-test
## Publish to pypi (using the API token stored in the PYPI_TEST_API_TOKEN_DATASCOPE environment variable).
publish-test:
	$(if $(PYPI_TEST_API_TOKEN_DATASCOPE),$(eval export TWINE_PASSWORD := $(PYPI_TEST_API_TOKEN_DATASCOPE)))
	twine upload --username __token__ --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
## Publish to pypi (using the API token stored in the PYPI_API_TOKEN_DATASCOPE environment variable).
publish:
	$(if $(PYPI_API_TOKEN_DATASCOPE),$(eval export TWINE_PASSWORD := $(PYPI_API_TOKEN_DATASCOPE)))
	twine upload --username __token__ dist/*

.PHONY: clean
## Remove the distribution files.
publish-clean:
	rm -rf dist/
	rm -rf datascope.egg-info

.PHONY: docs
## Generate the documentation.
docs:
	pdoc -d markdown --docformat numpy --output-dir docs --logo https://ease.ml/images/easeml-component-generic_hu85950f4ba52b697b532288a389d2b3d7_8523_250x250_fit_q100_h1_box_2.webp --logo-link https://ease.ml ./datascope