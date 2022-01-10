#!/usr/bin/env python3

import argparse
import experiments

from experiments import Experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the DataScope experiments.")

    domains = Experiment.domains
