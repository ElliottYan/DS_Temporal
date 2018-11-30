#!/usr/bin/env bash

cd ..

BATCH_SIZE=50
MODEL=MIML_CONV
MAX_HOPS=1
DEVICE=2
VERSION=1
N_EPOCHS=15
QUERY_TYPE=RELATION
DECAY_RATIO=0.95
lr=0.007

CUDA_VISIBLE_DEVICES=$DEVICE python trainer.py --model=$MODEL \
                                                --cuda \
                                                --position_embedding \
                                                --max_hops=$MAX_HOPS \
                                                --batch_size=$BATCH_SIZE \
                                                --mem_version=$VERSION \
                                                --max_epochs=$N_EPOCHS \
                                                --remove_origin_query \
                                                --query_type=$QUERY_TYPE \
                                                --lr=$lr \
                                                --decay_interval=1 \
                                                --decay_ratio=$DECAY_RATIO \
                                                --optimizer=adam
#                                                --debug
