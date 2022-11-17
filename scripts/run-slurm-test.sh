#!/usr/bin/env bash

CMD="python -m experiments run \
    --scenario label-repair \
    --dataset UCI \
    --method random \
    --pipeline identity \
    --providers 0 \
    --utility acc \
    --trainsize 100 \
    --valsize 50 \
    --testsize 50"

CMD+=" ${@}"

echo $CMD
eval $CMD