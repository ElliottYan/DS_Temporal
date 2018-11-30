#!/usr/bin/env bash

cd ..

BATCH_SIZE=50
MODEL=WORD_REL_MEM
MAX_HOPS=1
DEVICE=3
VERSION=1
N_EPOCHS=15
QUERY_TYPE=RELATION
DECAY_RATIO=0.5
#lr=0.002
#lr=0.00125
lr=0.001
#lr=0.001
#lr=0.00075
#lr=5e-4
#lr=2.5e-4

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
                                                --optimizer=adadelta
#                                                --use_noise_and_clip \
#                                                --debug
