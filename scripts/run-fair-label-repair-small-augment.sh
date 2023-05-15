#!/usr/bin/env bash

CMD="python -m datascope.experiments run \
    --scenario label-repair \
    --dataset FolkUCI \
    --pipeline std-scaler-mi-kmeans \
    --model logreg xgb \
    --method random shapley-knn-single shapley-knn-raw shapley-knn-interactive shapley-tmc-010 shapley-tmc-100 shapley-tmc-pipe-010 shapley-tmc-pipe-100 \
    --repairgoal fairness \
    --trainsize 1000 \
    --valsize 500 \
    --testsize 500 \
    --augment-factor 10 \
    --checkpoints 100 "

CMD+=" ${@}"

eval $CMD
