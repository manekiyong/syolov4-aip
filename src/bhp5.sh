#!/bin/bash
#python train.py --weights yolov4-p5_.pt --data data/bh.yaml --img 640 640 --device 0 --name p5_bh --epochs 100 --batch 8

python3 train.py \
    --device 0 \
    --weights "../models/pretrained/yolov4-p5_.pt" \
    --name "bh_p5" \
    --data "data/bh_s3.yaml" \
    --epochs 1 \
    --batch 8 \
    --img 640 640 \
    --nosave \
    --clearml \
    --s3 \
    --data_proj_name "datasets/yolov4/brainhack"