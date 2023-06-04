#!/usr/bin/env bash

CMD="python -m datascope.experiments report \
        --partby dataset pipeline \
        --index discarded_rel \
        --compare method \
        --targetval accuracy \
        --sliceby iteration \
        --sliceop max \
        --summarize importance_cputime \
        --plot line:accuracy bar:importance_cputime \
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