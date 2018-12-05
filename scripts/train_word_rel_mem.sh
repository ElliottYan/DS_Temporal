#!/usr/bin/env bash

cd ..

BATCH_SIZE=50
MODEL=WORD_REL_MEM
MAX_HOPS=1
DEVICE=0
VERSION=1
N_EPOCHS=15
QUERY_TYPE=RELATION
DECAY_RATIO=0.95
lr=0.01
word_mem_hops=1
rel_mem_hops=1
CONV_TYPE=CNN

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
                                                --optimizer=adam \
                                                --word_mem_hops=$word_mem_hops \
                                                --rel_mem_hops=$rel_mem_hops \
                                                --conv_type=$CONV_TYPE
#                                                --debug