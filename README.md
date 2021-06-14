# DataScope: Measuring data importance over ML pipelines using the Shapley value

This is a tool for inspecting ML pipelines by measuring how important is each training data point for predicting the label of a given test data example.

## Development Guide

### Setup

Assuming [pipenv](https://pipenv.pypa.io/) is installed, execute command: `pipenv install -e .`

### Repo Structure

This repository has several sections:

* Algorithms for computing the Shapley value are in `datascope/algorithms`
* Code for inspecting ML pipelines defined with sklearn are in `datascope/inspection`
* All tests are stored under `tests`
* Any experiment scripts and Jupyter notebooks are stored under `experiments`

### Adding Dependencies

For now we can keep all Python dependencies in the `requirements.txt`. Later on we will separate them by "flavor" (e.g. test dependencies, experiment dependencies).
