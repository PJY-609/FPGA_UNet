#!/usr/bin/env python
# coding: utf-8

import os
import time
from data_sequence import DataSequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import pathlib


def train(config):
    train_seq = DataSequence.from_table(config.TRAIN_FP, config.DATA_COL, config.TARGET_COL, config.NUM_CLASSES, config.BATCH_SIZE, augment=True)
    val_seq = DataSequence.from_table(config.VAL_FP, config.DATA_COL, config.TARGET_COL, config.NUM_CLASSES, config.BATCH_SIZE, augment=False)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=Config.reduce_lr_factor, patience=Config.reduce_lr_patience, cooldown=0, mode='min', verbose=1, min_lr=1e-8)
    early_stop = EarlyStopping(monitor='val_loss', patience=config.EARLYSTOP_PATIENCE, verbose=1, mode='min')
    ckpt_fp = os.path.join(config.CHECKPOINT_DIR, "model-{epoch:02d}-{val_loss:.2f}.hdf5")
    model_checkpoint = ModelCheckpoint(filepath=ckpt_fp, monitor='val_loss', save_weights_only=config.SAVE_WEIGHTS_ONLY, save_best_only=config.SAVE_BEST_ONLY, verbose=1, mode='min')
    csv_logger = CSVLogger(config.TRAIN_LOG_FP)
    callbacks = [early_stop, model_checkpoint, csv_logger]

    config.MODEL.summary()

    history = config.MODEL.fit_generator(generator=train_seq,
                        validation_data=val_seq,
                        steps_per_epoch=len(train_seq),
                        validation_steps=len(val_seq),
                        epochs=config.EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        shuffle=True
                        # workers=2
                        )
