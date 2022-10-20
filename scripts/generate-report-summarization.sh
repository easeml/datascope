#!/usr/bin/env bash

CMD="python -m experiments report \
        --groupby dataset pipeline \
        --index discarded_rel \
        --compare method \
        --targetval accuracy \
        --summarize importance_compute_time \
        --plot line:accuracy bar:importance_compute_time \
        --ylogscale false true \
        --aggmode median-perc-90 \
        --summode median-perc-90 \
        --labelformat \"%(method)s\" \
        --plotsize 6 5 \
        --legend true \
        --annotation true \
        --fontsize 22 \
        --dontcompare - random \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s\" "

        # --resultfilter \"method != 'shapley-knn-interactive'\" \

CMD+=" ${@}"

echo $CMD
eval $CMD