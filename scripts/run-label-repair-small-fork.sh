#!/usr/bin/env bash

CMD="python -m datascope.experiments run \
    --scenario label-repair \
    --method random shapley-knn-single shapley-knn-interactive shapley-tmc-010 shapley-tmc-100 shapley-tmc-pipe-010 shapley-tmc-pipe-100 \
    --repairgoal accuracy \
    --trainsize 1000 \
    --valsize 500 \
    --testsize 500 \
    --providers 100 \
    --dirtyratio 1.0"

CMD+=" ${@}"

eval $CMD