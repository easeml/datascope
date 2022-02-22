#!/usr/bin/env bash

METHOD=random shapley-knn-single shapley-knn-interactive shapley-tmc-pipe-010 shapley-tmc-pipe-100
python -m experiments run \
    --scenario data-discard \
    --dataset UCI \
    --method  ${METHOD} \
    --trainsize 1000 \
    --valsize 500 \
    --valbias 0.8
