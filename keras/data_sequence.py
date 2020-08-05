#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import Sequence
import pathlib
import pandas as pd
from utils import batch_standarization, load_masks, load_images
from data_augmentation import data_augment

class DataSequence(Sequence):
    def __init__(self, data_fps, target_fps, num_classes, n_samples, batch_size, augment):
        self.data_fps = data_fps
        self.target_fps = target_fps
        self.num_classes = num_classes
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.augment = augment
    
    @classmethod
    def from_table(cls, table_fp, data_col, target_col, num_classes, batch_size, augment):
        df = pd.read_csv(table_fp)
        data_fps =  df[data_col].values
        target_fps = df[target_col].values
        n_samples = len(df)
        return cls(data_fps, target_fps, num_classes, n_samples, batch_size, augment)

    @classmethod
    def from_folder(cls, data_dir, target_dir, num_classes, batch_size, augment):
        data_dir = pathlib.Path(data_dir)
        data_fps = [str(fp) for fp in data_dir.iterdir()]
        target_dir = pathlib.Path(target_dir)
        target_fps = [str(fp) for fp in target_dir.iterdir()]
        return cls(data_fps, target_fps, num_classes, len(data_fps), batch_size, augment) 

    def __len__(self):
        return int(np.ceil(self.n_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        
        dfps = self.data_fps[batch_slice]
        data_batch = load_images(dfps)

        tfps = self.target_fps[batch_slice]
        target_batch = load_masks(tfps, self.num_classes)

        if self.augment:
            data_batch, target_batch = data_augment(data_batch, target_batch)

        data_batch = batch_standarization(data_batch)
        return data_batch, target_batch
     
