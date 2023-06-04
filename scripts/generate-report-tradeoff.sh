#!/usr/bin/env bash

CMD="python -m datascope.experiments report \
        --partby dataset pipeline \
        --index steps \
        --compare method \
        --compareorder random shapley-knn-single shapley-tmc-pipe-010 shapley-tmc-pipe-100 shapley-tmc-010 shapley-tmc-100 \
        --summarize importance_cputime repaired_rel \
        --resultfilter \"accuracy_rel >= 0.9\" \
        --sliceby iteration \
        --plot dot:importance_cputime:repaired_rel \
        --xlogscale true \
        --aggmode median-perc-90 \
        --summode median-perc-90 \
        --labelformat \"%(method)s\" \
        --plotsize 6 5 \
        --legend true \
        --fontsize 22 \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s\" "

CMD+=" ${@}"

echo $CMD
eval $CMD