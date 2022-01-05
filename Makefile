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
	test -d $(VENV_DIR) || virtualenv $(VENV_DIR) --prompt "(datascope) " --python $(PYTHON)
	touch $(VENV_DIR)/touchfile

.PHONY: shell
## Load an instance of the shell with the appropriate virtual environment.
shell: $(VENV_DIR)/touchfile
#exec "bash -i <<< 'source $(VENV_DIR)/bin/activate; exec </dev/tty'"
# echo $(SHELL)
# exec "bash --init-file $(VENV_DIR)/bin/activate"
	. $(VENV_DIR)/bin/activate && exec bash

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
setup-exp: requirements-exp.txt
	pip install -e .[exp]

.PHONY: setup-all
## Install all bundles of datascope to the current python environment.
setup-all:
	pip install -e .[complete]

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
