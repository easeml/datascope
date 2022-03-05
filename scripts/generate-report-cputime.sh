#!/usr/bin/env bash

CMD="python -m experiments report \
        --compare method \
        --targetval importance_compute_time \
        --plot line:importance_compute_time \
        --xlogscale true \
        --ylogscale true \
        --aggmode median-perc-90 \
        --plotsize 6 5 \
        --titleformat \"\" \
        --fontsize 22 \
        --errdisplay bar \
        --legend false"

CMD+=" ${@}"

echo $CMD
eval $CMD
