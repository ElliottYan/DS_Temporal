#!/usr/bin/env bash

cd ..

BATCH_SIZE=4
MODEL=BERT_FINE_TUNE
PROBLEM=BERT_NYT_10
DEVICE=3
VERSION=1
N_EPOCHS=20
DECAY_RATIO=0.95
lr=0.005

CUDA_VISIBLE_DEVICES=$DEVICE python trainer.py --model=$MODEL \
                                                --cuda \
                                                --problem=$PROBLEM \
                                                --position_embedding \
                                                --batch_size=$BATCH_SIZE \
                                                --mem_version=$VERSION \
                                                --max_epochs=$N_EPOCHS \
                                                --lr=$lr \
                                                --decay_interval=1 \
                                                --decay_ratio=$DECAY_RATIO \
                                                --optimizer=adam
#                                                --debug
