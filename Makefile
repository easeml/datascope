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
	test -d $(VENV_DIR) || python -m venv $(VENV_DIR) --prompt "(datascope)" --python $(PYTHON)
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
	mypy --install-types --non-interactive

.PHONY: setup-exp
## Install the base and experimental bundles of datascope to the current python environment.
setup-exp: requirements-exp.txt
	pip install -e .[exp]
	pip install git+https://github.com/DS3Lab/datascope-pipelines

.PHONY: setup-all
## Install all bundles of datascope to the current python environment.
setup-all: requirements.txt requirements-dev.txt requirements-exp.txt
	pip install -e .[complete]
	pip install git+https://github.com/DS3Lab/datascope-pipelines

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

clean:
	rm -rf $(VENV_DIR)
	find -iname "*.pyc" -delete

.PHONY: package
## Package into .whl and source code archive (.tar.gz).
package:
	python -m build

.PHONY: publish-test
## Publish to pypi (test)
publish-test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish-test
## Publish to pypi
publish:
	twine upload dist/*

.PHONY: publish-clean
## remove the distribution files
publish-clean:
	rm -rf dist/
	rm -rf datascope.egg-info

.PHONY: docs
## Generate the documentation.
docs:
	pdoc -d markdown --docformat numpy --output-dir docs --logo https://ease.ml/images/easeml-component-generic_hu85950f4ba52b697b532288a389d2b3d7_8523_250x250_fit_q100_h1_box_2.webp --logo-link https://ease.ml ./datascope