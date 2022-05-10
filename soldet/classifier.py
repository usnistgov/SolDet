#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used for image classifier training.

Created on Sat May 29 16:09:34 2021
@author: sjguo
"""

import glob
import os
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from soldet.tf_helper import f1



def create_classifier(**model_params):
    '''
    Create blank CNN model with given model parameters. 
    If model parameters are not specified, generate default model decribed in
    arXiv: 2101.05404

    Parameters
    ----------
    model_params : dictionary
        model paramters, contains:
            #To be described#
            'kernel':
            'filters':
            'dense':
            'dropout':
            'loss':
            'opt':
                
            #To be added#
            'pool'
            'stride'
            
    Returns
    -------
    model : tf model
        Blank CNN model for classification.

    '''
    
    if 'kernel' in model_params:
        kernel = model_params['kernel']
    else:
        kernel = 5
        
    if 'filters' in model_params:
        filters = model_params['filters']
    else:
        filters=[8,16,32,64,128]
                
    if 'dense' in model_params:
        dense = model_params['dense']
    else:
        dense = [256,128,64]
            
    if 'dropout' in model_params:
        dropout = model_params['dropout']
    else:
        dropout = 0.5
        
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
            opt=tf.keras.optimizers.Adamax(learning_rate=0.001)
    else:
        opt=tf.keras.optimizers.Adamax(learning_rate=0.001)
    
    if 'loss' in model_params:
        loss = model_params['loss']
    else:
        loss = 'categorical_crossentropy'
    
    pools = 2
    strides = 2
    
    model = Sequential()
    model.add(InputLayer(input_shape=(132,164,1)))
    for fil in filters:
        model.add(Conv2D(filters=fil, kernel_size=[kernel, kernel], padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(pools, pools), strides=strides))
    model.add(Flatten())
    for den in dense:
        model.add(Dense(units=den, activation='relu'))
        model.add(Dropout(rate=dropout))
    model.add(Dense(units=3, activation='softmax'))
   
    model.compile(optimizer=opt,
                      loss=loss,
                      metrics=[f1,'acc'])
    return model


def cross_validate(data_list, labels_list, **model_params):
    '''
    Do a X-fold cross validation, return average weighted F1 score.
    length of data_list and labels_list decides X.
    
    By default, use balance_by_flip and ImageDataGenerator 
    # Need description / Way to turn off
    
    Parameters
    ----------
    data_list : list of 3-D numpy arrays.
        A list of labeled images. Dimension N x 132 x 164.
    labels_list : list of 2-D numpy arrays. dtype=int.
        A list of labels with length N.
    **model_params : dictionary
        model/training paramters, beside ones described in create_model:
            'save': string. Saving directory. If same model exist in this directory,
                    output evaluation without training.
                    e.g. "Data/models"
            
            'batch_size': batch size for training
            'epoch': epoch for training
            'common_data' & 'common_labels': data and label used in all of folds.
            
            #To be added#
            'augment_params':

    Returns
    -------
    XV_result : Float
        average weighted F1 score.

    '''
    # training parameters:
    if ('common_data' in model_params) and ('common_labels' in model_params):
        common_data_flag = True
        common_data = model_params['common_data']
        common_labels = model_params['common_labels']
    else:
        common_data_flag = False
        
    if 'batch_size' in model_params:
        batch_size = model_params['batch_size']
    else:
        batch_size = 32
    
    if 'epoch' in model_params:
        epoch = model_params['epoch']
    else:
        epoch = 30
    
    if 'save' in model_params:
        # Reading model parameters for saving name
        if 'kernel' in model_params:
            kernel = model_params['kernel']
        else:
            kernel = 5
                    
        if 'filters' in model_params:
            filters = model_params['filters']
        else:
            filters=[8,16,32,64,128]
                        
        if 'dense' in model_params:
            dense = model_params['dense']
        else:
            dense = [256,128,64]
            
        fil=''
        for j in filters:
            fil+=str(j)+'_'
    
        den=''
        for j in dense:
            den+=str(j)+'_'
    
        if 'dropout' in model_params:
            dropout = model_params['dropout']
        else:
            dropout = 0.5
        
        if 'opt' in model_params:
            opt = model_params['opt']
        else: 
            opt = 'Adamax'
        
        #TODO: add loss?
        
        save = True
        save_dir = model_params['save'] + "/fil_"+fil+'den_'+den+"ker_"+str(kernel)+"_drop_"+str(dropout)+"_opt_"+opt+"_batch_"+str(batch_size)+"_epoch_"+str(epoch)+"_"
        trained_models = glob.glob(model_params['save'] + "/*.h5")
    else:
        save = False
        save_dir = '1'
        trained_models = []
    
    x_val = len(data_list)
    bal_data_list = []
    bal_label_list = []
    for i in range(x_val):
        b_data, b_labels = balance_data_by_flip(data_list[i], labels_list[i])
        bal_data_list.append(b_data)
        bal_label_list.append(b_labels)
    
    results = []
    if common_data_flag:
        b_c_data, b_c_labels = balance_data_by_flip(common_data, common_labels)
        
    for i in range(x_val):
        
        # test data
        test_data = data_list[i]
        test_labels = labels_list[i]
                
        # if choosen params have trained, import model
        if save and ((save_dir+str(i)+".h5") in trained_models):
            model = load_model(glob.glob(os.getcwd()+save_dir+str(i)+".h5")[0], custom_objects={'f1': f1})
            
        # otherwise, create model and train
        else:
            model = create_classifier(**model_params)
            
            # train data
            train_data = np.concatenate([x for j,x in enumerate(bal_data_list) if j!=i])
            train_labels = np.concatenate([x for j,x in enumerate(bal_label_list) if j!=i])
            if common_data_flag:
                train_data = np.append(train_data, b_c_data, axis=0)
                train_labels = np.append(train_labels, b_c_labels, axis=0)
                
            train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)

            # training 
            datagen = ImageDataGenerator(rotation_range=1, fill_mode='wrap')
            datagen.fit(train_data)
            it = datagen.flow(train_data, train_labels, batch_size=batch_size)
            model.fit(it, steps_per_epoch=int(np.floor(train_data.shape[0]/batch_size)), epochs=epoch)
        
        # save to h5
        if save:
            model.save(save_dir + str(i) + ".h5")
        
        # evaluate and return average f1 of cross validation
        ev = model.evaluate(test_data,test_labels)
        results.append(ev[1])
    
    XV_result = np.mean(results)
    return XV_result



def label_4_to_x(label_in, x=3):
    '''
    Merge 4 labels to x labels (x=2,3): 
    e.g.1. x=3 [0,3,1,2,3] -> [0,2,1,2,2]
    e.g.2. x=3 [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] ->
    [[1,0,0],[0,1,0],[0,0,1],[0,0,1]] 
    
    Parameters
    ----------
    label_in : 1-D or 2-D numpy array
    x: int, 2 or 3
    
    Returns
    -------
    res : 1-D or 2-D numpy array

    '''
    
    label_shape = label_in.shape
    
    if len(label_shape) == 1:
        label_out = []
        m = x-1
        for l in label_in:
            label_out.append(min(l,m))
        return np.array(label_out)
    
    if len(label_shape) == 2:
        if label_shape[1] !=4:
            print("2-D label must have length 4, input label has shape:", label_shape)
            return 0
        
        else:
            label_out = np.zeros((label_shape[0],x))
            if x==2:
                for i in range(label_shape[0]):
                    if label_in[i,1] == 1:
                        label_out[i,1] = 1
                    else:
                        label_out[i,0] = 1
            elif x==3:
                for i in range(label_shape[0]):
                    if label_in[i,1] == 1:
                        label_out[i,1] = 1
                    elif label_in[i,0] == 1:
                        label_out[i,0] = 1
                    else:
                        label_out[i,2] = 1
            else:
                print("x must be 2 or 3, input x is:", x)
                return 0
            
            return label_out
    
    else:
        print('label_in must be 1-D or 2-D.')
        

def balance_data_by_flip(labeled_data_list, labels_list):
    '''
    Balance data by flips. If label = 0,2, filp in all 3 ways + original image. 
    If label == 1, choose 1 of 3 random filp + original image.
    3 ways:
        (1) Horizontal flip
        (2) Vertical flip
        (3) Both Horizontal and Vertical flip

    Parameters
    ----------
    labeled_data_list : 3-D numpy array
        A list of labeled images. Dimension N x 132 x 164.
    labels_list : 1-D numpy array dtype=int
        A list of labels with length N.

    Returns
    -------
    balanced_data_list : 3-D numpy array
        A list of labeled images. Dimension M x 132 x 164. M is the number of 
        balanced data.
    balanced_labels_list :1-D numpy array dtype=int
        A list of labels with length M.

    '''
    
    l = len(labeled_data_list)
    if len(labels_list) == l:
        new_data = []
        new_labels = []
        
        a = np.random.randint(3, size=l)
        
        for i in range(l):
            new_data.append(labeled_data_list[i])
            new_labels.append(labels_list[i])
            if labels_list[i,1]==1:
                if a[i]==0:
                    new_data.append(np.flipud(labeled_data_list[i]))
                elif a[i]==1:
                    new_data.append(np.fliplr(labeled_data_list[i]))
                else:
                    new_data.append(np.flipud(np.fliplr(labeled_data_list[i])))
                new_labels.append(labels_list[i])
            else:
                new_data.append(np.flipud(labeled_data_list[i]))
                new_labels.append(labels_list[i])
                new_data.append(np.fliplr(labeled_data_list[i]))
                new_labels.append(labels_list[i])
                new_data.append(np.flipud(np.fliplr(labeled_data_list[i])))
                new_labels.append(labels_list[i])
        balanced_data_list = np.array(new_data)
        balanced_labels_list = np.array(new_labels)
        return balanced_data_list, balanced_labels_list 
    else:
        print("data and labels not equal size")
        return 0

