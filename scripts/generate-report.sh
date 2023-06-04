#!/usr/bin/env bash

if [ -z "$2" ]
then
    TARGETVAL=accuracy
else
    TARGETVAL=$2
fi

if [ -z "$1" ]
then
    python -m datascope.experiments report \
        --partby dataset pipeline utility \
        --index steps \
        --compare method \
        --targetval ${TARGETVAL} \
        --aggmode median-perc-95 \
        --summode mean-std \
        --summarize importance_cputime \
        --labelformat "%(method)s - %(importance_cputime:mean).2f +/- %(importance_cputime:std).2f"

else
    python -m datascope.experiments report \
        --partby dataset pipeline utility \
        --index steps \
        --compare method \
        --targetval ${TARGETVAL} \
        --aggmode median-perc-95 \
        --summode mean-std \
        --summarize importance_cputime \
        --labelformat "%(method)s - %(importance_cputime:mean).2f +/- %(importance_cputime:std).2f" \
        --study-path "$1" \
        --output-path "$1/reports/${TARGETVAL}"
fi