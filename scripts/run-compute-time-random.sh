#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario compute-time \
    --dataset random \
    --pipeline identity \
    --method shapley-knn-single \
    --trainsize 10000 100000 1000000 \
    --valsize 1000"

CMD+=" ${@}"

echo "Target Variable: Training Set Size"
echo $CMD
eval $CMD

CMD="python -m experiments run \
    --scenario compute-time \
    --dataset random \
    --pipeline identity \
    --method shapley-knn-single \
    --trainsize 100000 \
    --valsize 100 1000 10000"

CMD+=" ${@}"

echo "Target Variable: Validation Set Size"
echo $CMD
eval $CMD

CMD="python -m experiments run \
    --scenario compute-time \
    --dataset random \
    --pipeline identity \
    --method shapley-knn-single \
    --trainsize 100000 \
    --valsize 1000 \
    --numfeatures 10 100 1000"

CMD+=" ${@}"

echo "Target Variable: Number of features"
echo $CMD
eval $CMD