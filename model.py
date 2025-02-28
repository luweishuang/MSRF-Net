import os
import re
import csv
import json
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import warnings
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input,Average,Conv2DTranspose,SeparableConv2D,dot,UpSampling2D,Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D,Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D
from keras import backend as K
from keras.layers import concatenate ,Lambda
import itertools
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint 
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50,VGG19,VGG16,DenseNet121,DenseNet169,InceptionResNetV2
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy
import numpy as np
from skimage.morphology import square,binary_erosion,binary_dilation,binary_opening,binary_closing
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras import regularizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from math import sqrt, ceil
from PIL import Image
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
from glob import glob
import tifffile as tif
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import *
from keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import *
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def dice_coefficient_loss(y_true, y_pred):
    return 1.-dice_coefficient(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice(y_true, y_pred):
    smooth = 1.
    w, m1, m2 = 0.7, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return K.sum(score)


def weighted_dice_loss(y_true, y_pred):
    smooth = 1.
    w, m1, m2 = 0.7, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    d1 =  (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return d1


def spatial_att_block(x,intermediate_channels):
    out = Conv2D(intermediate_channels,kernel_size=(1,1),strides=(1,1),padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='same')(out)
    out = Activation('sigmoid')(out)
    return out


def resblock(x,ip_channels,op_channels,stride=(1,1)):
    residual = x
    out = Conv2D(op_channels,kernel_size=(3,3),strides=stride,padding='same')(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(op_channels,kernel_size=(3,3),strides=stride,padding='same')(x)
    out = BatchNormalization()(out)
    out = Add()([out,residual])
    out = Activation('relu')(out)
    return out


def dual_att_blocks(skip,prev,out_channels):
    up = Conv2DTranspose(out_channels,4, strides=(2, 2), padding='same')(prev)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    inp_layer = Concatenate()([skip,up])
    inp_layer = Conv2D(out_channels,3,strides=(1,1),padding='same')(inp_layer)
    inp_layer = BatchNormalization()(inp_layer)
    inp_layer = Activation('relu')(inp_layer)
    se_out = se_block(inp_layer,out_channels)
    sab = spatial_att_block(inp_layer,out_channels//4)
    #sab = Add()([sab,1])
    sab = Lambda(lambda y : y+1)(sab)
    final = Multiply()([sab,se_out])
    return final
    

def gsc(input_features,gating_features,in_channels,out_channels,kernel_size=1,stride=1,dilation=1,groups=1):
    x = Concatenate()([input_features,gating_features])
    x = BatchNormalization()(x)
    x = Conv2D(in_channels+1, (1,1), strides =(1,1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(1,kernel_size=(1,1),strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    #x = sigmoid(x)
    return x


def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return Multiply()([in_block, x])


def Attention_B(X, G, k):
    FL = int(X.shape[-1])
    init = RandomNormal(stddev=0.02)
    theta = Conv2D(k,(2,2), strides = (2,2), padding='same')(X)
    Phi = Conv2D(k, (1,1), strides =(1,1), padding='same', use_bias=True)(G)
   
    ADD = Add()([theta, Phi])
   
    #ADD = LeakyReLU(alpha=0.1)(ADD)
    ADD = Activation('relu')(ADD)
   
    #Psi = Conv2D(FL,(1,1), strides = (1,1), padding="same",kernel_initializer=init)(ADD)
    Psi = Conv2D(1,(1,1), strides = (1,1), padding="same",kernel_initializer=init)(ADD)
    Psi = Activation('sigmoid')(Psi)
    Up = Conv2DTranspose(1, (2,2), strides=(2, 2), padding='valid')(Psi)
   
    #Psi = Activation('tanh')(Psi)
   
    #Up = Conv2DTranspose(FL, (2,2), strides=(2, 2), padding='valid')(Psi)
   
    Final = Multiply()([X, Up])
    #Final = Conv2D(1, (1,1), strides = (1,1), padding="same",kernel_initializer=init)(Final)
    #Final = Conv2D(FL, (1,1), strides = (1,1), padding="same",kernel_initializer=init)(Final)
    Final = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-5)(Final)
    print(Final.shape)
    return Final


def Unet3(input_shape,n_filters,kernel=(3,3),strides=(1,1),pad='same'):
    x = input_shape
    conv1 = Conv2D(n_filters,kernel_size=kernel,strides=strides,padding=pad)(input_shape)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
   
    conv2 = Conv2D(n_filters,kernel_size=kernel,strides=strides,padding=pad)(conv1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 =  LeakyReLU(alpha=0.1)(conv2)
   
    x = Conv2D(n_filters,kernel_size = (1,1),strides = (1,1),padding = 'same')(x)
   
    return Add()([x,conv2])


def Up3(input1,input2,kernel=(3,3),stride=(1,1), pad='same'):
    #up = UpSampling2D(2)(input2)
    up = Conv2DTranspose(int(input1.shape[-1]),(1, 1), strides=(2, 2), padding='same')(input2)
    up = Concatenate()([up,input1])
    
    #up1 = BatchNormalization()(up)
    #up1 =  LeakyReLU(alpha=0.25)(up1)
    #up1 = Conv2D(int(input1.shape[-1]),kernel_size=(3,3),strides=(1,1),padding='same')(up1)
    #up1 = BatchNormalization()(up1)
    #up1 =  LeakyReLU(alpha=0.25)(up1)
    #up1 = Conv2D(int(input1.shape[-1]),kernel_size=(3,3),strides=(1,1),padding='same')(up1)
    #up2 = Add()([up1,up])
    return up
    
    return Unet3(up,int(input1.shape[-1]),kernel,stride,pad)


def gatingSig(input_shape,n_filters,kernel=(1,1),strides=(1,1),pad='same'):
    conv = Conv2D(n_filters,kernel_size=kernel,strides=strides,padding=pad)(input_shape)
    conv = BatchNormalization(axis=-1)(conv)
    return LeakyReLU(alpha=0.1)(conv)


def DSup(x, var):
    d = Conv2D(1,(1,1), strides=(1,1), padding = "same")(x)
    d = UpSampling2D(var)(d)
    return d


def DSup1(x, var):
    d = Conv2D(1,(2,2), strides=(2,2), padding = "same")(x)
    d = UpSampling2D(var)(d)
    return d

#Keras
ALPHA = 0.5
BETA = 0.5
GAMMA=1


def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        return 1 - Tversky


def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = K.pow((1 - Tversky), gamma)
        
        return FocalTversky


def msrf(input_size=(256,256,3),input_size_2=(256,256,1)):
    n_labels=1
    feature_scale=8
    #input_shape= Image.shape
    filters = [64, 128, 256, 512,1024]
    atrous_rates = (6, 12, 18)
    n_labels=1
    feature_scale=8
    #input_shape= Image.shape
    filters = [64, 128, 256, 512,1024]

    inputs_img = Input(input_size)
    canny = Input(input_size_2,name='checkdim')
    n11 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_img)
    n11 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n11)
    n11 = BatchNormalization()(n11)
    n11 = se_block(n11,32)

    
    n12 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n11)
    n12 = BatchNormalization()(n12)
    #here
    n12 = Add()([n12,n11])
    pred1 = Conv2D(1,(1,1), strides=(1,1), padding="same",activation='sigmoid')(n12)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(n11)
    pool1 = Dropout(0.2)(pool1)
    n21 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    n21 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n21)
    n21 = BatchNormalization()(n21)
    n21 = se_block(n21,64)
    #here
    
    
    n22 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n21)
    n22 = BatchNormalization()(n22)
    n22 = Add()([n22,n21])
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(n21)
    pool2 = Dropout(0.2)(pool2)
    
    n31 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    n31 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n31)
    n31 = BatchNormalization()(n31)
    n31 = se_block(n31,128)
   
    
    n32 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n31)
    n32 = BatchNormalization()(n32)
    n32 = Add()([n32,n31])
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(n31)
    pool3 = Dropout(0.2)(pool3)
    n41 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    n41 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n41)
    n41 = BatchNormalization()(n41)
    #############################################ASPP
    shape_before = n41.shape
    
    
    
    #############################################
    
    n12,n22 = RDDB(n11,n21,32,64,16)
    pred2 = Conv2D(1,(1,1), strides=(1,1), padding="same",activation='sigmoid')(n12)
    
    n32,n42 = RDDB(n31,n41,128,256,64)
    
    n12,n22 = RDDB(n12,n22,32,64,16)
    pred3 = Conv2D(1,(1,1), strides=(1,1), padding="same",activation='sigmoid')(n12)
    
    n32,n42 = RDDB(n32,n42,128,256,64)
    
    n22,n32 = RDDB(n22,n32,64,128,32)
    
    
    n13,n23 = RDDB(n12,n22,32,64,16)
    
    n33,n43 = RDDB(n32,n42,128,256,64)

    n23,n33 = RDDB(n23,n33,64,128,32)

    n13,n23 = RDDB(n13,n23,32,64,16)
    
    n33,n43 = RDDB(n33,n43,128,256,64)

    n13 = Lambda(lambda x: x * 0.4)(n13)
    n23 = Lambda(lambda x: x * 0.4)(n23)
    n33 = Lambda(lambda x: x * 0.4)(n33)
    n43 = Lambda(lambda x: x * 0.4)(n43)

   
    n13,n23 = Add()([n11,n13]),Add()([n21,n23])
    n33,n43 = Add()([n31,n33]),Add()([n41,n43])

    ###############Shape Stream

    
    d0 = Conv2D(32,kernel_size=(1,1),strides=(1,1),padding='same')(n23)
    ss = keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(d0)
    ss = resblock(ss,32,32)
    c3 = Conv2D(1, kernel_size=(1,1),strides=(1,1),padding='same')(n33)
    c3 = keras.layers.UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(c3)
    ss = Conv2D(16,kernel_size=(1,1),strides=(1,1),padding='same')(ss)
    ss = gsc(ss,c3,32,32)
    ss = resblock(ss,16,16)
    ss = Conv2D(8,kernel_size=(1,1),strides=(1,1),padding='same')(ss)
    c4 = Conv2D(1, kernel_size=(1,1),strides=(1,1),padding='same')(n43)
    c4 = keras.layers.UpSampling2D(size=(8, 8), data_format=None, interpolation='bilinear')(c4)
    ss = gsc(ss,c4,16,16)
    ss = resblock(ss,8,8)
    ss = Conv2D(4,kernel_size=(1,1),strides=(1,1),padding='same')(ss)
    ss = Conv2D(1,kernel_size=(1,1),padding='same')(ss)
    edge_out = Activation('sigmoid',name='edge_out')(ss)
    #######canny edge
    #canny = cv2.Canny(np.asarray(inputs),10,100)
    cat = Concatenate()([edge_out,canny])
    cw = Conv2D(1,kernel_size=(1,1),padding='same')(cat)
    acts = Activation('sigmoid')(cw)
    edge = Conv2D(1, kernel_size=(1,1),strides=(1,1),padding='same')(acts)
    edge = BatchNormalization()(edge)
    edge = Activation('relu')(edge)
    
    
    #########################
    
   
    n34_preinput=Attention_B(n33,n43,128)
    n34 = Up3(n34_preinput,n43)
    n34_d = dual_att_blocks(n33,n43,128)
    n34_t = Concatenate()([n34,n34_d])
    n34_t = Conv2D(128,kernel_size=(1,1),strides=(1,1),padding='same')(n34_t)
    n34_2 = BatchNormalization()(n34_t)
    n34_2 = Activation('relu')(n34_2)
    n34_2 = Conv2D(128,kernel_size=(3,3),strides=(1,1),padding='same')(n34_2)
    n34_2 = BatchNormalization()(n34_2)
    n34_2 = Activation('relu')(n34_2)
    n34_2 = Conv2D(128,kernel_size=(3,3),strides=(1,1),padding='same')(n34_2)
    n34 = Add()([n34_2,n34_t])
    pred4 = Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='same',activation="sigmoid")(n34)
    pred4 = UpSampling2D(size=(4,4),interpolation='bilinear',name='pred4')(pred4)

    
   
    n24_preinput =Attention_B(n23,n34,64)
    n24 = Up3(n24_preinput,n34)
    n24_d = dual_att_blocks(n23,n34,64)
    n24_t = Concatenate()([n24,n24_d])
    n24_t = Conv2D(64,kernel_size=(1,1),strides=(1,1),padding='same')(n24_t)
    n24_2 = BatchNormalization()(n24_t)
    n24_2 = Activation('relu')(n24_2)
    n24_2 = Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same')(n24_2)
    n24_2 = BatchNormalization()(n24_2)
    n24_2 = Activation('relu')(n24_2)
    n24_2 = Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same')(n24_2)
    n24 = Add()([n24_2,n24_t])
    pred2 = Conv2D(1,kernel_size=(1,1),strides=(1,1),padding="same" , activation="sigmoid")(n24)
    pred2 = UpSampling2D(size=(2,2),interpolation='bilinear',name='pred2')(pred2)
   
    n14_preinput = Conv2DTranspose(32,4, strides=(2, 2), padding='same')(n24)
    n14_input = Concatenate()([n14_preinput,n13])
    n14_input = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n14_input)
    n14 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n14_input)
    n14 = BatchNormalization()(n14)
    n14 = Add()([n14,n14_input])
    n14 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(n14)
    x = Conv2D(1,(1,1), strides=(1,1), padding="same",activation='sigmoid',name='x')(n14)
    model = Model(inputs= [inputs_img,att],outputs = [x,pred2,pred4,pred8])
    return model


def RDDB(x,y,nf1=128,nf2=1212,gc=64,bias=True):
    x1 = Conv2D(filters=gc, kernel_size=3, strides=1,padding='same', bias=bias)(x)
    x1 = LeakyReLU(alpha=0.25)(x1)
    
    y1 = Conv2D(filters=gc, kernel_size=3, strides=1,padding='same', bias=bias)(y)
    y1 = LeakyReLU(alpha=0.25)(y)
    
    x1c = Conv2D(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(x)
    x1c = LeakyReLU(alpha=0.25)(x1c)
    y1t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(y)
    y1t = LeakyReLU(alpha=0.25)(y1t)
    
    
    x2_input = concatenate([x,x1,y1t],axis=-1)
    x2 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same',bias=bias)(x2_input)
    x2 = LeakyReLU(alpha=0.25)(x2)
    
    y2_input = concatenate([y,y1,x1c],axis=-1)
    y2 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same',bias=bias)(y2_input)
    y2 = LeakyReLU(alpha=0.25)(y2)
    
    x2c = Conv2D(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(x1)
    x2c = LeakyReLU(alpha=0.25)(x2c)
    y2t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(y1)
    y2t = LeakyReLU(alpha=0.25)(y2t)
    
    
    
    x3_input = concatenate([x,x1,x2,y2t] , axis=-1)
    x3 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same', bias=bias)(x3_input)
    x3 = LeakyReLU(alpha=0.25)(x3)
    
    y3_input = concatenate([y,y1,y2,x2c] , axis=-1)
    y3 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same', bias=bias)(y3_input)
    y3 = LeakyReLU(alpha=0.25)(y3)
    
    x3c = Conv2D(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(x3)
    x3c = LeakyReLU(alpha=0.25)(x3c)
    y3t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(y3)
    y3t = LeakyReLU(alpha=0.25)(y3t)
    
    
        
    x4_input = concatenate([x,x1,x2,x3,y3t] , axis=-1)
    x4 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same', bias=bias)(x4_input)
    x4 = LeakyReLU(alpha=0.25)(x4)
    
    
    y4_input = concatenate([y,y1,y2,y3,x3c] , axis=-1)
    y4 = Conv2D(filters= gc, kernel_size=3,strides=1, padding='same', bias=bias)(y4_input)
    y4 = LeakyReLU(alpha=0.25)(y4)
    
    x4c = Conv2D(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(x4)
    x4c = LeakyReLU(alpha=0.25)(x4c)
    y4t = Conv2DTranspose(filters=gc, kernel_size=3, strides=2,padding='same', bias=bias)(y4)
    y4t = LeakyReLU(alpha=0.25)(y4t)
    
        
    x5_input = concatenate([x,x1,x2,x3,x4,y4t] , axis=-1)
    x5 = Conv2D(filters= nf1, kernel_size=3,strides=1, padding='same', bias=bias)(x5_input)
    x5 = LeakyReLU(alpha=0.25)(x5)
    
    y5_input = concatenate([y,y1,y2,y3,y4,x4c] , axis=-1)
    y5 = Conv2D(filters= nf2, kernel_size=3,strides=1, padding='same', bias=bias)(y5_input)
    y5 = LeakyReLU(alpha=0.25)(y5)
        
    x5 = Lambda(lambda x: x * 0.4)(x5)
    y5 = Lambda(lambda x: x * 0.4)(y5)
        
    return Add()([x5,x]),Add()([y5,y])
