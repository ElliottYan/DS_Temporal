#!/usr/bin/env bash

BATCH_SIZE=160
MODEL=MEM_CNN
MAX_HOPS=1
DEVICE=2
VERSION=1
N_EPOCHS=25
QUERY_TYPE=RELATION

CUDA_VISIBLE_DEVICES=$DEVICE python trainer.py --model=$MODEL\
                                                --cuda \
                                                --position_embedding \
                                                --max_hops=$MAX_HOPS \
                                                --batch_size=$BATCH_SIZE \
                                                --mem_version=$VERSION \
                                                --max_epochs=$N_EPOCHS \
                                                --remove_origin_query \
                                                --query_type=$QUERY_TYPE
