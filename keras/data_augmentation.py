#!/usr/bin/env python
# coding: utf-8

import numpy as np
import imgaug.augmenters as iaa


def augment():
    return iaa.SomeOf((0, 4), [
            # iaa.Add((-50, 50)),
            iaa.CropAndPad(percent=(-0.4, 0.4)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
                       translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                       rotate=(-90, 90)),
            iaa.SomeOf((1, 3), [
                iaa.Dropout(p=(0.01, 0.2)),
                iaa.GaussianBlur(sigma=(0.0, 1.5)),
                iaa.AverageBlur(k=(3, 7)),
                # iaa.MedianBlur(k=(3, 7)),
                iaa.MotionBlur(k=(3, 7)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                iaa.SaltAndPepper(p=0.2),
                iaa.GammaContrast((0.5, 2.0))
                ])
            ], random_order=True)


def data_augment(images, masks):
    aug = augment()
    images, masks = aug.augment(images=images, segmentation_maps=masks)
    return images, masks



