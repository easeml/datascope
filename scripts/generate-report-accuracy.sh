#!/usr/bin/env bash

CMD="python -m experiments report \
        --groupby dataset pipeline \
        --index repaired_rel \
        --compare method \
        --compareorder random shapley-knn-single shapley-knn-interactive shapley-tmc-010 shapley-tmc-100 \
        --targetval accuracy \
        --summarize importance_cputime \
        --sliceby iteration \
        --sliceop max \
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

CMD+=" ${@}"

echo $CMD
eval $CMD