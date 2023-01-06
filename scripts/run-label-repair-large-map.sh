#!/usr/bin/env bash

CMD="python -m datascope.experiments run \
    --scenario label-repair \
    --dataset FolkUCI \
    --repairgoal accuracy \
    --method random shapley-knn-single \
    --trainsize 0 \
    --valsize 1000 \
    --testsize 1000"

CMD+=" ${@}"

echo $CMD
eval $CMD