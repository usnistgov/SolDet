#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used for object detector training.

Created on Sat May 29 16:10:32 2021
@author: sjguo
"""

import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import backend as K
from soldet.mhat_metric import find_soliton
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_obj_detector(**model_params):
    '''
    Create blank object detector model with given model parameters.

    Parameters
    ----------
    model_params : dictionary
        model paramters, contains:
            #To be described#
            'kernel':
            'filters':
            'opt':
                
            #To be added#
            'pool'
            'stride'
            
    Returns
    -------
    model : tf model
        Blank CNN model for object detection.

    '''
    
    if 'kernel' in model_params:
        kernel = model_params['kernel']
    else:
        kernel = 5
        
    if 'filters' in model_params:
        filters = model_params['filters']
    else:
        filters=[8,16,32,64,128]
        
    if 'opt' in model_params:
        if model_params['opt']=='SGD':
            opt=tf.keras.optimizers.SGD(learning_rate=0.01)
        elif model_params['opt']=='SGDM':
            opt=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        elif model_params['opt']=='SGDMD':
            opt=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.01)
        elif model_params['opt']=='Adam':
            opt=tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            opt=tf.keras.optimizers.Adamax(learning_rate=0.001, clipnorm=1.)
    else:
        opt=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.)
    
    # pools = 2
    
    model = Sequential()
    model.add(InputLayer(input_shape=(132,164,1)))
    
    model.add(Conv2D(filters=8, kernel_size=[5, 5], padding="same", activation='relu', activity_regularizer='l1_l2'))
    # model.add(Conv2D(filters=8, kernel_size=[7, 7], padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(Conv2D(filters=16, kernel_size=[5, 5], padding="same", activation='relu', activity_regularizer='l1_l2'))
    # model.add(Conv2D(filters=16, kernel_size=[7, 7], padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation='relu', activity_regularizer='l1_l2'))
    model.add(MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding="same"))
    model.add(Conv2D(filters=64, kernel_size=[1, 5], padding="same", activation='relu', activity_regularizer='l1_l2'))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="same"))
    model.add(Conv2D(filters=128, kernel_size=[1, 5], padding="same", activation='relu', activity_regularizer='l1_l2'))
    model.add(Conv2D(filters=2, kernel_size=[1, 5], padding="same", activation='sigmoid'))
    model.compile(optimizer=opt,
                  loss=Metz_loss,
                  metrics=[f1_41])
    return model

def labels_to_positions(data, labels):
    '''
    find list of positions,
    position = [deepest positions] if labels=1, position = [] if labels=0

    Parameters
    ----------
    data : numpy array
        DESCRIPTION.
    labels : numpy array or list
        soliton number labels, only 0 and 1 are allowed.

    Returns
    -------
    positions : list
        list of positions.

    '''
    
    positions = []
    l = len(data)
    for i in range(l):
        if labels[i]==0:
            positions.append([])
        elif labels[i]==1:
            positions.append([find_soliton(data[i])['cen']])
        else:
            print('Error: Labelc not only contains 0 and 1: label = ', labels[i])
    return positions

def filter_augment_position(raw_data, raw_pos, seed=None):
    ind = []
    pos = []
    for i,p in enumerate(raw_pos):
        if type(p) is list:
            ind.append(i)
            pos.append(p)
    data = raw_data[ind]
    data, pos = augment_data_by_flip(data, pos, seed=seed)
    pos = pos_41labels_conversion(pos)
    return data, pos

def augment_data_by_flip(labeled_data_list, pos_list, seed=None):
    l = len(labeled_data_list)
    if len(pos_list) == l:
        new_data = []
        new_labels = []
                
        for i in range(l):
            new_data.append(labeled_data_list[i])
            new_data.append(np.fliplr(labeled_data_list[i]))
            new_data.append(np.flipud(labeled_data_list[i]))
            new_data.append(np.flipud(np.fliplr(labeled_data_list[i])))

            if len(pos_list[i])==0:
                for j in range(4):
                    new_labels.append(pos_list[i])
            elif len(pos_list[i])==1:
                for j in range(2):
                    new_labels.append(pos_list[i])
                    new_labels.append([164 - pos_list[i][0]])
        if seed == None:
            s = random.randint(0, 1e7)
            random.Random(s).shuffle(new_data)
            random.Random(s).shuffle(new_labels)
        else:
            random.Random(seed).shuffle(new_data)
            random.Random(seed).shuffle(new_labels)
        return np.array(new_data), new_labels 
    else:
        print("data and labels not equal size")
        return

def pos_41labels_conversion(label_in, threshold=[0.5, 4]):
    '''
    Convert between soliton position(s) and 1 x 41 x 2 object detector label.
    First 41 entrys represents possibility of soliton exist in each 4 x 132 cell
    Second 41 entrys represents scaled soliton positions in each cell, if exist.
    
    Parameters
    ----------
    label_in : List or np array
        DESCRIPTION.
    threshold: TYPE: list of two floats
        1. Threshold of probability to if soliton. default: 0.5
        2. Threshold of pixel distance between two solitons to merge. default: 4
    Returns
    -------
    None.

    '''
    label_out =[]
    if type(label_in) == list: # if input is soliton positions
        if label_in == []:
            label_out = np.zeros((1,41,2))
        elif type(label_in[0]) in [float, np.float64]: # Postions on Single image 
            label_out = np.zeros((1,41,2))
            for l in label_in:
                if l < 164 and l > 0:
                    label_out[0, int(l // 4), 0] = 1
                    label_out[0, int(l // 4), 1] = (l % 4)/4
                else:
                    print('soliton positon beyond [0, 164].')
                    
        elif type(label_in[0]) == list: # A list of postions on many images
            label_out = np.zeros((len(label_in),1,41,2))
            for i, pos in enumerate(label_in):
                for l in pos:
                    if l < 164 and l > 0:
                        label_out[i, 0, int(l // 4), 0] = 1
                        label_out[i, 0, int(l // 4), 1] = (l % 4)/4
                    else:
                        print('soliton positon beyond [0, 164].')
                
    elif type(label_in) == np.ndarray: # if input is 41 labels
        if label_in.shape == (1, 41, 2): # Single 41 label
            for i in range(41):
                if label_in[0, i, 0] > threshold[0]:
                    label_out.append(4 * i + 4 * label_in[0, i, 1])
            if len(label_out)>1:
                i = 0
                while (i+1)<len(label_out):
                    if (label_out[i+1] - label_out[i]) < threshold[1]:
                        label_out[i] = (label_out[i+1] + label_out[i])/2
                        del label_out[i+1]
                    else:
                        i +=1
        elif label_in.shape[1:] == (1, 41, 2):# Array of 41 labels
            for label in label_in:
                l_out = []
                for i in range(41):
                    if label[0, i, 0] > threshold[0]:
                        l_out.append(4 * i + 4 * label[0, i, 1])
                if len(l_out)>1:
                    i = 0
                    while (i+1)<len(l_out):
                        if (l_out[i+1] - l_out[i]) < threshold[1]:
                            l_out[i] = (l_out[i+1] + l_out[i])/2
                            del l_out[i+1]
                        else:
                            i +=1
                label_out.append(l_out)
        else:
            print('label(s) does not have shape (1, 41, 2).')
    else:
        print('label_in is neither list nor numpy array.')
    
    return label_out


def Metz_loss(y_true, y_pred):
    '''
    Loss function defined in arXiv: 2012.13097

    '''
    return K.sum(-10 * y_true[...,0] * K.log(y_pred[...,0] + K.epsilon()) 
                  - (1 - y_true[...,0]) * K.log(1 - y_pred[...,0] + K.epsilon())
                  + 10 * y_true[...,0] * K.square(y_true[...,1] - y_pred[...,1]))
  
  
def f1_41(true, pred):
    '''
    Return f1 score for 41 labels for detections. This metric is used for training.
    To see the f1 score for true/detected soliton positions use "f1_41merged"

    Parameters
    ----------
    true : 2-D numpy array dtype=int
        A list of true labels with shape (n_sample, n_class)
    pred : 2-D numpy array dtype=int
        A list of true labels with shape (n_sample, n_class)

    Returns
    -------
    weighted_f1 : float

    '''
    
    ground_positives = K.sum(true[...,0])       # = TP + FN
    pred_positives = K.sum(pred[...,0])         # = TP + FP
    true_positives = K.sum(true[...,0] * pred[...,0])  # = TP

    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon()) 
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon()) 
        #both = 1 if ground_positives == 0 or pred_positives == 0

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1


def _pos_41labels_conversion_keras(label_in):
    '''
    Simpler version of pos_41labels_conversion, for f1_41merged use
    '''
    label_out = []
    for label in label_in:
        l_out = []
        for i in range(41):
            if label[0, i, 0] > 0.5:
                l_out.append(4 * i + 4 * label[0, i, 1])
            if len(l_out)>1:
                for i in range(len(l_out)-1):
                    if (l_out[i+1] - l_out[i]) < 4:
                        l_out[i] = (l_out[i+1] + l_out[i])/2
                        del l_out[i+1]
            label_out.append(l_out)
    return label_out


def f1_41merged(true, pred):
    '''
    Return f1 score for true positive is defined as detetected soliton is within
    plus-minus 3 pixel range of a true soliton.
    

    Parameters
    ----------
    true : 2-D numpy array dtype=int
        A list of true labels with shape (n_sample, n_class)
    pred : 2-D numpy array dtype=int
        A list of true labels with shape (n_sample, n_class)

    Returns
    -------
    f1 : float

    '''
    true_pos = _pos_41labels_conversion_keras(true)
    pred_pos = _pos_41labels_conversion_keras(pred)
    ground_positives = 0
    pred_positives = 0
    true_positives = 0
    for i in range(len(true_pos)):
        ground_positives += len(true_pos[i])
        pred_positives +=len(pred_pos[i])
        for t in true_pos[i]:
            for p in pred_pos[i]:
                if abs(t-p) < 3:
                    true_positives += 1
    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon()) 
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon()) 
        #both = 1 if ground_positives == 0 or pred_positives == 0

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1

def f1_merged(true, pred):
    '''
    Return f1 score for true positive is defined as detetected soliton is within
    plus-minus 3 pixel range of a true soliton.
    

    Parameters
    ----------
    true : list of positions
        
    pred : list of positions

    Returns
    -------
    f1 : float

    '''
    ground_positives = 0
    pred_positives = 0
    true_positives = 0
    for i in range(len(true)):
        ground_positives += len(true[i])
        pred_positives +=len(pred[i])
        for t in true[i]:
            for p in pred[i]:
                if abs(t-p) < 3:
                    true_positives += 1
    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon()) 
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon()) 
        #both = 1 if ground_positives == 0 or pred_positives == 0

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1


