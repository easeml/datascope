#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario label-repair \
    --method random shapley-knn-single \
    --trainsize 0 \
    --valsize 1000 \
    --testsize 1000 \
    --providers 100 \
    --dirtyratio 1.0"

CMD+=" ${@}"

echo $CMD
eval $CMD