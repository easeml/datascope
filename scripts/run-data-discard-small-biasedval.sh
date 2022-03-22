#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario data-discard \
    --dataset UCI FolkUCI \
    --method random shapley-knn-single shapley-knn-interactive shapley-tmc-pipe-010 shapley-tmc-pipe-100 \
    --trainsize 1000 \
    --valsize 500 \
    --testsize 500 \
    --valbias 0.8"

CMD+=" ${@}"

echo $CMD
eval $CMD