#!/usr/bin/env bash

if [ -z "$2" ]
then
    TARGETVAL=accuracy
else
    TARGETVAL=$2
fi

if [ -z "$1" ]
then
    python -m experiments report \
        --groupby dataset pipeline utility \
        --index steps \
        --compare method \
        --targetval ${TARGETVAL} \
        --aggmode median-perc-95 \
        --summode mean-std \
        --summarize importance_compute_time \
        --labelformat "%(method)s - %(importance_compute_time:mean).2f +/- %(importance_compute_time:std).2f"

else
    python -m experiments report \
        --groupby dataset pipeline utility \
        --index steps \
        --compare method \
        --targetval ${TARGETVAL} \
        --aggmode median-perc-95 \
        --summode mean-std \
        --summarize importance_compute_time \
        --labelformat "%(method)s - %(importance_compute_time:mean).2f +/- %(importance_compute_time:std).2f" \
        --study-path "$1" \
        --output-path "$1/reports/${TARGETVAL}"
fi