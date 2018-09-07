#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:47:24 2017
@author: risabh
"""

from keras.models import  Model
from keras.layers import Reshape,  Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, cv2
import glob
from utils.utils import decode_netout, draw_boxes, scale_boxes

# Hide Tensorflow Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print('Please Wait ...')


LABELS = ['BCT']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
THRESHOLD        = 0.30
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 0.5
OBJECT_SCALE     = 1.0
COORD_SCALE      = 5.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 2
WARM_BUP_BATCH   = 0
TRUE_BOX_BUFFER  = 50


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

# Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False, input_shape=(416,416,3))(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])

# Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)
# Layer 23
x = Conv2D((4 + 1 + CLASS) * 5, (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)

#Model whose weight to load
model.load_weights("model/weights_yolo_adam_validdata_temp.h5")

dummy_array = np.zeros((1,1,1,1,50,4))

#File path for test images
filePath = input('>>> Enter Image Folder Path: ')
if filePath[-1] != '/':
    filePath += '/'

#Path to save the generated results
fileOut =  './result/detect_ref/'
ImageOut = './result/image/'
#File path for corresponding annotation for the images
annoOut = './result/boxes/'

fileDir = [im for im in os.listdir(filePath) if im.endswith(".png")]
print('Reading Images: ', fileDir)

try:
    os.stat('./result/')
except:
    os.mkdir('./result/')   

try:
    os.stat(fileOut)
except:
    os.mkdir(fileOut)   

try:
    os.stat(ImageOut)
except:
    os.mkdir(ImageOut)

try:
    os.stat(annoOut)
except:
    os.mkdir(annoOut)


iou_global = []
iou_final =0.0
counter =0

for f in fileDir:
    counter+=1
    print('Detecting ',f , '...', end='   ')
    annot_fname = str(f)[:-4]+'.txt'
   
    image = cv2.imread(filePath+ f)
    
    cv2.imwrite(ImageOut + f, image)  # Original Image Output
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    
    netout = model.predict([input_image, dummy_array])
    
    boxes = decode_netout(netout[0], 
                          obj_threshold=0.25, #predicted box_score should be greater than this threshold 
                          nms_threshold=0.28, #threshold limiting the percentage operlap between the predicted bounding boxes
                          anchors=ANCHORS, 
                          nb_class=CLASS)
    # Boxes has x,y,w,h 
    boxes = scale_boxes(image, boxes)

    print('Got ',len(boxes), ' cells!')
    
    #if gorund truth boxes are required to be shown in image pass gt=gt below
    image_box, final_boxes = draw_boxes(image, boxes, labels=LABELS, h_threshold=0.2 , w_threshold=0.2, h_min=0.05, w_min=0.05,gt=[]) 

    cv2.imwrite(fileOut + f, image)

    with open(annoOut+annot_fname, 'w') as pred_file:
        for eachpred in final_boxes:
            pred_file.write('%.2f %.2f %.2f %.2f\n' % (eachpred[0], eachpred[1], eachpred[2], eachpred[3]))
