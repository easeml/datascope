#!/usr/bin/env bash

CMD="python -m datascope.experiments report \
        --partby model dataset pipeline utility \
        --index repaired_rel \
        --xtickfmt percent \
        --compare method \
        --compareorder random shapley-knn-single shapley-knn-interactive shapley-tmc-pipe-010 shapley-tmc-pipe-100 shapley-tmc-010 shapley-tmc-100 \
        --colors random:blue shapley-knn-single:red shapley-knn-interactive:yellow shapley-tmc-pipe-010:green shapley-tmc-pipe-100:brown shapley-tmc-010:purple shapley-tmc-100:cyan \
        --targetval accuracy eqodds \
        --sliceby iteration \
        --sliceop max \
        --summarize importance_cputime \
        --plot line:accuracy line:eqodds bar:importance_cputime \
        --ylogscale false false true \
        --aggmode median-perc-90 \
        --summode median-perc-90 \
        --labelformat \"%(method)s\" \
        --plotsize 5 4 \
        --annotation true \
        --fontsize 22 \
        --no-multiprocessing \
        --usetex false \
        --legend false \
        --dontcompare - - random \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s; Model: %(model)s\" \"Optimize for: %(utility)s\" "

CMD+=" ${@}"

echo $CMD
eval $CMD