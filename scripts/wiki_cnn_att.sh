#!/usr/bin/env bash

cd ..

BATCH_SIZE=160
MODEL=CNN_ATT
MAX_HOPS=3
DEVICE=0
VERSION=1
N_EPOCHS=50
QUERY_TYPE=RELATION
DECAY_RATIO=0.5
#lr=0.001
#lr=0.00125
#lr=0.00075
#lr=5e-4
lr=0.01
DATA_ROOT=./data

CUDA_VISIBLE_DEVICES=$DEVICE python trainer.py --model=$MODEL \
                                                --dataset_dir=$DATA_ROOT \
                                                --cuda \
                                                --position_embedding \
                                                --order_embed \
                                                --max_hops=$MAX_HOPS \
                                                --batch_size=$BATCH_SIZE \
                                                --mem_version=$VERSION \
                                                --max_epochs=$N_EPOCHS \
                                                --remove_origin_query \
                                                --query_type=$QUERY_TYPE \
                                                --problem=WIKI-TIME \
                                                --lr=$lr \
                                                --decay_ratio=$DECAY_RATIO \
                                                --use_noise_and_clip
#                                                --circular
