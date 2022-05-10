#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to assist TensorFlow 2.2

Created on Tue Jun  1 16:35:29 2021
@author: sjguo
"""

import random
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import sklearn
from .object_detector import pos_41labels_conversion

def cross_validate(data, labels, model_creator, fold=5, augmentor=False, **cv_params):
    if ('common_data' in cv_params) and ('common_labels' in cv_params):
        if len(cv_params['common_data'])==len(cv_params['common_labels']):
            common_data_flag = True
            common_data = cv_params['common_data']
            common_labels = cv_params['common_labels']
        else:
            print('common_data and common_labels does not have same length.')
            return
    else:
        common_data_flag = False
        
    if 'batch_size' in cv_params:
        batch_size = cv_params['batch_size']
    else:
        batch_size = 32
    
    if 'epochs' in cv_params:
        epochs = cv_params['epochs']
    else:
        epochs = 30

    if 'save' in cv_params:
        save = True
        save_name = cv_params['save']
    else:
        save = False

    if 'return_fold_index' in cv_params:
        return_fold_index = cv_params['return_fold_index']
    else:
        return_fold_index = False
    
    if 'use_ImageDataGenerator' in cv_params: #TODO: add later
        pass
    
    if 'only_train_fold' in cv_params: # Dev tool
        only_train_fold = cv_params['only_train_fold']
    else:
        only_train_fold = False
    
    # split data and labels
    data_list = np.array_split(data, fold)
    labels_list = np.array_split(labels, fold)
    
    # adding augmentations / datagenerator / common data to training set
    train_data_list = []
    train_labels_list = []
    
    if augmentor is not False:
        if common_data_flag:
            common_data, common_labels = augmentor(common_data, common_labels)
        for i in range(fold):
            aug_data, aug_labels = augmentor(data_list[i], labels_list[i])
            train_data_list.append(aug_data)
            train_labels_list.append(aug_labels)
    else:
        train_data_list = data_list
        train_labels_list = labels_list
    
    pred = []
    if type(only_train_fold) is not list:
        train_fold = list(range(fold))
    else:
        train_fold = only_train_fold
    
    for i in train_fold:
        train_data = np.concatenate([x for j,x in enumerate(train_data_list) if j!=i])
        train_labels = np.concatenate([x for j,x in enumerate(train_labels_list) if j!=i])
        if common_data_flag: #ADD common data
            train_data = np.append(train_data, common_data, axis=0)
            train_labels = np.append(train_labels, common_labels, axis=0)
        test_data = train_data_list[i]
        test_labels = train_labels_list[i]
        
        ev = 0
        while ev < 0.8: # make model until it is a good one.
            # shuffle data
            train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)
            
            # training
            model = model_creator()
            model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
            ev = model.evaluate(test_data, test_labels)[1] # Need to be f1
            
        # save trained model
        if save:
            model.save(save_name + str(i) + ".h5")
        
        # save output prediected results
        pred.append(model.predict(data_list[i]))
        
    pred = np.concatenate(pred)
    
    if return_fold_index:
        fold_id = []
        for i in range(fold):
            fold_id.append(np.ones((len(labels_list[i])))*i)
        fold_id = np.concatenate(fold_id)
        return pred, fold_id
    else:
        return pred

def f1(true, pred):
    '''
    Return weighted f1 score between multi-class true labels and prediction

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
    
    ground_positives = K.sum(true, axis=0)       # = TP + FN
    pred_positives = K.sum(pred, axis=0)         # = TP + FP
    true_positives = K.sum(true * pred, axis=0)  # = TP

    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon()) 
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon()) 
        #both = 1 if ground_positives == 0 or pred_positives == 0

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1 # for loss, can return 1 - weighted_f1


def label_vec_num_conversion(labels):
    '''
    Convert bewteen 1-D number labels and 2-D vector labels.
    e.g. [0,0,1,2] <-> [[1,0,0],[1,0,0],[0,1,0],[0,0,1]]

    Parameters
    ----------
    labels : 1-D or 2-D numpy array

    Returns
    -------
    res : 2-D or 1-D numpy array

    '''
    
    shape_labels=labels.shape
    if len(shape_labels) ==1:
        res = np.zeros((shape_labels[0],len(np.unique(labels))))
        for i,l in enumerate(labels):
            res[i,int(l)] = 1

    elif len(shape_labels) ==2:
        res = np.zeros((shape_labels[0]), dtype=int)
        for i,l in enumerate(labels):
            res[i] =  int(list(l).index(1))
    else:
        print("Input label array is neither 1D nor 2D.")
    return res

def pred_to_labels(prob):
    '''
    Take probablistic prediected results to classifciation result.
    e.g. [0.5, 0.25, 0.25] -> [1, 0, 0]
    
    Parameters
    ----------
    prob : 2-D numpy array
        DESCRIPTION.

    Returns
    -------
    labels : 2-D numpy array
        DESCRIPTION.

    '''
    prob = np.array(prob)
    
    labels = np.zeros(prob.shape, dtype=int)
    for i in range(prob.shape[0]):
        labels[i,np.argmax(prob[i])]=1
    return labels


def plot_histroy(history): # need improve: single image, input string
    '''
    Plot training history

    Parameters
    ----------
    history : Tensorflow histroy object.

    Returns
    -------
    None.

    '''
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for lossZ
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
