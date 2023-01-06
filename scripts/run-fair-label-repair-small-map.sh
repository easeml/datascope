#!/usr/bin/env bash

CMD="python -m datascope.experiments run \
    --scenario label-repair \
    --method random shapley-knn-single shapley-knn-interactive shapley-tmc-pipe-010 shapley-tmc-pipe-100 \
    --repairgoal fairness \
    --trainsize 1000 \
    --valsize 500 \
    --testsize 500"

CMD+=" ${@}"

eval $CMD
