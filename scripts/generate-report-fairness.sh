#!/usr/bin/env bash

CMD="python -m experiments report \
        --groupby dataset pipeline utility \
        --index discarded_rel \
        --compare method \
        --targetval accuracy eqodds \
        --summarize importance_compute_time \
        --plot line:accuracy line:eqodds bar:importance_compute_time \
        --ylogscale false false true \
        --aggmode median-perc-90 \
        --summode median-perc-90 \
        --labelformat \"%(method)s\" \
        --plotsize 6 5 \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s; Optimize for: %(utility)s\" "

CMD+=" ${@}"

echo $CMD
eval $CMD