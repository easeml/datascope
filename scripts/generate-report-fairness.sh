#!/usr/bin/env bash

CMD="python -m experiments report \
        --groupby dataset pipeline utility \
        --index repaired_rel \
        --compare method \
        --targetval accuracy eqodds \
        --sliceby iteration \
        --sliceop max \
        --summarize importance_cputime \
        --plot line:accuracy line:eqodds bar:importance_cputime \
        --ylogscale false false true \
        --aggmode median-perc-90 \
        --summode median-perc-90 \
        --labelformat \"%(method)s\" \
        --plotsize 6 5 \
        --legend true \
        --annotation true \
        --fontsize 22 \
        --dontcompare - - random \
        --titleformat \"Dataset: %(dataset)s; Pipeline: %(pipeline)s; Optimize for: %(utility)s\" "

CMD+=" ${@}"

echo $CMD
eval $CMD