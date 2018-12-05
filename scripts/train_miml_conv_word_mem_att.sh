#!/usr/bin/env bash

cd ..

BATCH_SIZE=50
MODEL=MIML_CONV_WORD_MEM_ATT
MAX_HOPS=1
WORD_MEM_HOPS=2
DEVICE=1
VERSION=1
N_EPOCHS=10
QUERY_TYPE=RELATION
DECAY_RATIO=0.95
#lr=0.007
lr=0.002
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
                                                --word_mem_hops=$WORD_MEM_HOPS \
                                                --decay_interval=1 \
                                                --decay_ratio=$DECAY_RATIO \
                                                --optimizer=adam \
                                                --conv_type=$CONV_TYPE \
                                                --use_whole_bag
#                                                --debug
