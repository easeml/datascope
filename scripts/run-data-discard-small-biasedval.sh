#!/usr/bin/env bash

CMD="python -m datascope.experiments run \
    --scenario data-discard \
    --dataset UCI FolkUCI \
    --model logreg knn xgb \
    --method random shapley-knn-single shapley-knn-interactive shapley-tmc-010 shapley-tmc-100 shapley-tmc-pipe-010 shapley-tmc-pipe-100 \
    --repairgoal accuracy \
    --utility acc \
    --trainsize 1000 \
    --valsize 500 \
    --testsize 500 \
    --trainbias 0.8 \
    --valbias 0.5 \
    --biasmethod label \
    --maxremove 0.9 "

CMD+=" ${@}"

echo $CMD
eval $CMD