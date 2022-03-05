#!/usr/bin/env bash

CMD="python -m experiments report \
        --groupby dataset pipeline \
        --index repaired_rel \
        --compare method \
        --targetval accuracy \
        --summarize importance_compute_time \
        --plot line:accuracy bar:importance_compute_time \
        --ylogscale false true \
        --aggmode median-perc-90 \
        --summode median-perc-90 \
        --labelformat \"%(method)s\" \
        --plotsize 6 5 \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s\" "

CMD+=" ${@}"

echo $CMD
eval $CMD