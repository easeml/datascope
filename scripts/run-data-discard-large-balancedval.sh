#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario data-discard \
    --dataset FolkUCI \
    --method random shapley-knn-single \
    --repairgoal fairness \
    --trainsize 0 \
    --valsize 1000 \
    --testsize 1000 \
    --trainbias 0.8"

CMD+=" ${@}"

echo $CMD
eval $CMD
