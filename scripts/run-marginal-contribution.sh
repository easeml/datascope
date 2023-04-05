#!/usr/bin/env bash

CMD="python -m datascope.experiments run \
    --scenario marginal-contribution \
    --dataset UCI FolkUCI FashionMNIST TwentyNewsGroups DataPerfVision CifarN \
    --pipeline identity std-scaler log-scaler mi-kmeans pca pca-svd tf-idf tolower-urlremove-tfidf gauss-blur hog-transform \
    --utility acc rocauc \
    --model logreg knn-1 knn-5 knn-10 knn-50 \
    --trainsize 1000 \
    --testsize 500 "

CMD+=" ${@}"

eval $CMD
