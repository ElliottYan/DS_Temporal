#!/usr/bin/env bash

cd ..

BATCH_SIZE=160
MODEL=CONV_AVE
MAX_HOPS=1
DEVICE=1
VERSION=1
N_EPOCHS=20
QUERY_TYPE=RELATION
DECAY_RATIO=0.95
lr=0.00125
CONV_TYPE=PCNN

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
                                                --optimizer=sgd \
                                                --conv_type=$CONV_TYPE
#                                                --debug
