#!/usr/bin/env bash

CMD="python -m experiments report \
        --groupby dataset pipeline \
        --index repaired_rel \
        --compare method \
        --compareorder random shapley-knn-single shapley-tmc-pipe-010 shapley-tmc-pipe-100 shapley-tmc-010 shapley-tmc-100 \
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
        --dontcompare shapley-knn-interactive random,shapley-knn-interactive \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s\" "

CMD+=" ${@}"

echo $CMD
eval $CMD