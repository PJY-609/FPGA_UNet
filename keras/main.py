#!/usr/bin/env python
# coding: utf-8

from train import train
from model import unet2d


class Config:
    # data
    TRAIN_FP = '/home/juezhao/FPGA_UNet/train.csv'
    VAL_FP = '/home/juezhao/FPGA_UNet/val.csv'
    DATA_COL = 'img'
    TARGET_COL = 'msk'
    NUM_CLASSES = 2

    # training
    BATCH_SIZE = 16
    EPOCHS = 2000
    LR = 1e-3
    DROP_RATE = 0.3
    MODEL = unet2d(input_size=(288, 384, 3), num_classes=NUM_CLASSES, lr=LR, drop_rate=DROP_RATE)

    # callbacks
    EARLYSTOP_PATIENCE = 50
    REDUCE_LR_PATIENCE = 2000
    REDUCE_LR_FACTOR = 0.2
    TRAIN_LOG_FP = 'train_log.txt'
    CHECKPOINT_DIR = '.'
    SAVE_WEIGHTS_ONLY = True
    SAVE_BEST_ONLY = True


if __name__ == '__main__':
    train(Config)
