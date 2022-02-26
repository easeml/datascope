#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario label-repair \
    --method random shapley-knn-single shapley-knn-interactive shapley-tmc-pipe-010 shapley-tmc-pipe-100 \
    --trainsize 0 \
    --valsize 1000 \
    --providers 100 \
    --dirtyratio 1.0"

if [ "$1" ]
then
    CMD+=" -o $1"
fi

eval $CMD