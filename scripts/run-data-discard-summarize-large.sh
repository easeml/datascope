#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario data-discard \
    --discardgoal summarization \
    --method random shapley-knn-single \
    --trainsize 0 \
    --valsize 500 \
    --maxremove 0.9"

CMD+=" ${@}"

echo $CMD
eval $CMD
