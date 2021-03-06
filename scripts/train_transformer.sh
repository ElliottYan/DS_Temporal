#!/usr/bin/env bash

cd ..

BATCH_SIZE=50
MODEL=TRANSFORMER_ENCODER
DEVICE=2
VERSION=1
N_EPOCHS=20
QUERY_TYPE=RELATION
DECAY_RATIO=0.95
lr=0.001
d_model=128
d_ff=256
heads=1
num_layers=1

CUDA_VISIBLE_DEVICES=$DEVICE python trainer.py --model=$MODEL \
                                                --cuda \
                                                --position_embedding \
                                                --batch_size=$BATCH_SIZE \
                                                --mem_version=$VERSION \
                                                --max_epochs=$N_EPOCHS \
                                                --remove_origin_query \
                                                --query_type=$QUERY_TYPE \
                                                --lr=$lr \
                                                --decay_interval=1 \
                                                --decay_ratio=$DECAY_RATIO \
                                                --optimizer=sgd \
                                                --d_model=$d_model \
                                                --num_layers=$num_layers \
                                                --heads=$heads \
                                                --d_ff=$d_ff \
                                                --random_embedding
#                                                --use_whole_bag \
#                                                --debug
