#!/usr/bin/env bash

CMD="python -m experiments report \
        --compare method \
        --targetval importance_cputime \
        --index trainsize \
        --plot line:importance_cputime \
        --xlogscale true \
        --ylogscale true \
        --xtickfmt engineer \
        --aggmode median-perc-90 \
        --plotsize 6 5 \
        --titleformat \"\" \
        --fontsize 28 \
        --errdisplay bar \
        --legend false"

CMD+=" ${@}"

echo $CMD
eval $CMD
