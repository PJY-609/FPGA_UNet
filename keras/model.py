#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet2d(pretrained_weights=None, input_size=(256,256,1), num_classes=2, lr=1e-4, drop_rate=0.):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    drop1 = Dropout(drop_rate)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    

    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    drop2 = Dropout(drop_rate)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    
    
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    drop3 = Dropout(drop_rate)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    

    conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    drop4 = Dropout(drop_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


    conv5 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    drop5 = Dropout(drop_rate)(conv5)

    up6 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    up6 = BatchNormalization()(up6)
    up6 = Activation('relu')(up6)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    drop6 = Dropout(drop_rate)(conv6)

    up7 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = BatchNormalization()(up7)
    up7 = Activation('relu')(up7)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    drop7 = Dropout(drop_rate)(conv7)

    up8 = Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    up8 = BatchNormalization()(up8)
    up8 = Activation('relu')(up8)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    drop8 = Dropout(drop_rate)(conv8)

    up9 = Conv2D(32, 2, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = BatchNormalization()(up9)
    up9 = Activation('relu')(up9)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1)(conv10)
    conv10 = Activation('softmax')(conv10)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# def unet3d(pretrained_weights=None,input_size=(128,128,128,1)):
#     inputs = Input(input_size)
#     conv1 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1a')(inputs)
#     conv1 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1b')(conv1)
#     pool1 = MaxPooling3D(pool_size=(2,2,2), name='pool1')(conv1)
#     conv2 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2a')(pool1)
#     conv2 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2b')(conv2)
#     pool2 = MaxPooling3D(pool_size=(2,2,2), name='pool2')(conv2)
#     conv3 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3a')(pool2)
#     conv3 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3b')(conv3)
#     pool3 = MaxPooling3D(pool_size=(2,2,2), name='pool3')(conv3)
#     conv4 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4a')(pool3)
#     conv4 = Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4b')(conv4)
    
#     up5 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up-conv5')(UpSampling3D(size = (2,2,2), name='up5')(conv4))
#     merge5 = concatenate([conv3,up5], axis=-1)
#     conv5 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5a')(merge5)
#     conv5 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5b')(conv5)

#     up6 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up-conv6')(UpSampling3D(size = (2,2,2), name='up6')(conv5))
#     merge6 = concatenate([conv2,up6], axis=-1)
#     conv6 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6a')(merge6)
#     conv6 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6b')(conv6)

#     up7 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='up-conv7')(UpSampling3D(size=(2,2,2))(conv6))
#     merge7 = concatenate([conv1,up7], axis=-1)
#     conv7 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7a')(merge7)
#     conv7 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7b')(conv7)

#     conv8 = Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8')(conv7)
#     conv9 = Conv3D(1, 1, activation = 'sigmoid', name='conv9')(conv8)

#     model = Model(input=inputs, output=conv9)

#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
#     model.summary()

#     if(pretrained_weights):
#         model.load_weights(pretrained_weights)

#     return model

