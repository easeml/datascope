#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario label-repair \
    --method random shapley-knn-single \
    --trainsize 0 \
    --valsize 1000"

CMD+=" ${@}"

echo $CMD
eval $CMD