#!/usr/bin/env python3

import argparse
import timeit

from dspipes import Pipelines
from Experiment import Experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the DataScope experiments.")

    DATASETS = ["FashionMNIST", "UCI", "20NewsGroups"]
    parser.add_argument("-d", "--datasets", type=str, nargs='+', default=DATASETS, choices=DATASETS,
        help="Datasets to use. Default: [all datasets]")

    parser.add_argument("-i", "--iterations", type=int, default=1000,
        help="Number of iterations used in Monte-Carlo methods. Default: 1000.")

    parser.add_argument("-o", "--output-path", type=str, default="./",
        help="Path where the data is stored. Default: [current directory]")

    EXPERIMENTS = ["labels", "fairness", "poisoning"]
    parser.add_argument("-e", "--experiments", type=str, nargs='+', default=EXPERIMENTS, choices=EXPERIMENTS,
        help="Space-separated list of experiments to run. Default: [all experiments]")

    FORKSETS = [0, 10, 100]
    parser.add_argument("-f", "--forksets", type=int, nargs='+', default=FORKSETS, choices=FORKSETS,
        help="Space-separated list of forkset sizes to run. Default: [all sizes]")
    
    ALL_PIPELINES = list(range(10))
    DATASET_PIPELINES = {
        "FashionMNIST" : list(range(0, 8)),
        "UCI" : list(range(0, 6)),
        "20NewsGroups" : list(range(8, 9))
    }
    parser.add_argument("-p", "--pipelines", nargs='+', default=ALL_PIPELINES, choices=ALL_PIPELINES,
        help="Space-separated list of pipelines to run. Default: [all pipelines]")

    args = parser.parse_args()

    total_start = timeit.default_timer()

    for dataset in args.datasets:

        print("Dataset: %s / [%s]" % (dataset, ", ".join(args.datasets)))

        for forkset in args.forksets:

            print("Forkset: %s / [%s]" % (forkset, ", ".join(str(f) for f in args.forksets)))

            pipelines = list(set(DATASET_PIPELINES[dataset]) & set(args.pipelines))

            for pipeline in pipelines:

                print("Pipeline: %d / [%s]" % (pipeline, ", ".join(str(p) for p in pipelines)))
                print()

                print("Running experiment for:")
                print("  - Dataset: %s" % dataset)
                print("  - Forkset: %s" % forkset)
                print("  - Pipeline: %d" % pipeline)
                print("  - Experiments: [%s]" % ", ".join(args.experiments))
                print()

                settings = {
                    'iterations': args.iterations, # number of MC iterations
                    'run_label' : "labels" in args.experiments,
                    'run_fairness' : "fairness" in args.experiments,
                    'run_poisoning': "poisoning" in args.experiments, # what experiment 
                    'ray': True, # use ray?
                    'truncated': True, # TMC or MC?
                    'run_forks': forkset # use forks? how many?
                }

                start = timeit.default_timer()

                name = f'pipe_{pipeline}'
                model = Pipelines.create_numerical_pipeline(name, imputer=False)
                exp_name = f'i{args.iterations}_{dataset}_{name}'
                exp = Experiment(exp_name, model, dataset_name=dataset, save_path=args.output_path)
                exp.run(**settings)

                stop = timeit.default_timer()
                print('Time: ', stop - start)

    total_stop = timeit.default_timer()
    print('Total runtime: ', total_stop - total_stop)
