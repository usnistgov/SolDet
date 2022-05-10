#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:41:47 2022

@author: sjguo
"""
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
import glob

import numpy as np
from tqdm import tqdm
import csv
#%%
# models_results = np.load(nb_dir+'/data/data_info/cross_validation_results.npy', allow_pickle=True).item()


# #

# models_results_new = {}

# for k, v in models_results.items():
    
#     models_results_new[k.replace('labeled_data/3/','labeled_data/2/').replace('/data/labeled_data/','/data/data_files/class-')] = v
    
# np.save(nb_dir+'/data/data_info/cross_validation_results', models_results_new)



#%% remove data information

labeled_data_dir = [nb_dir + '/data/data_files/class-' + s +'/' for s in ['0','1','2','8']]

files = []
for d in labeled_data_dir:
    files += glob.glob(d + "*.npy")
    
#%%

keys = ['label_batch_num', 'label_SG', 'label_JZ', 'label_old', 'label_SG_old', 'label_JZ_old', 'label_AF', 'label_Ian', 'question_flag_SG', 'question_flag_JZ']

keys_2 = ['positions', 'types']

csv_labeling_info = []
csv_labeling_info.append(['File names']+keys)

# csv_labeling_info_2 = []
# csv_labeling_info_2.append(['File names']+keys_2)

j=0
for file in tqdm(files):
    f = file.replace(nb_dir,'')
    data_dict = np.load(file, allow_pickle=True).item()
    csv_info = [f]
    for k in keys:
        if k in data_dict.keys():
            csv_info.append(data_dict[k])
            del data_dict[k]
        else:
            csv_info.append('')
    csv_labeling_info.append(csv_info)
    
    # if 'class-8' in f:
    #     csv_info_2 = [f]
    #     for k in keys_2:
    #         if k in data_dict.keys():
    #             csv_info_2.append(data_dict[k])
    #             del data_dict[k]
    #         else:
    #             csv_info_2.append('')
    #     csv_labeling_info_2.append(csv_info_2)

    np.save(file, data_dict) 
    
with open(nb_dir+'/data/data_info/6257_data_label_process.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_labeling_info)
    
# with open(nb_dir+'/data/data_info/159_data_mislabel_pos_type.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(csv_labeling_info_2)


#%%

labeled_data_dir = [nb_dir + '/data/data_files/class-' + s +'/' for s in ['8']]

files = []
for d in labeled_data_dir:
    files += glob.glob(d + "*.npy")
    
keys = ['label']

csv_labeling_info = []
csv_labeling_info.append(['File names']+keys)

for file in tqdm(files):
    f = file.replace(nb_dir,'')
    data_dict = np.load(file, allow_pickle=True).item()
    csv_info = [f]
    for k in keys:
        if k in data_dict.keys():
            csv_info.append(data_dict[k])
            del data_dict[k]
        else:
            csv_info.append('')
    csv_labeling_info.append(csv_info)
    np.save(file, data_dict) 
    
with open(nb_dir+'/data/data_info/879_misdata_previous_labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_labeling_info)
    
    

#%%

labeled_data_dir = [nb_dir + '/data/data_files/class-' + s +'/' for s in ['0','1','2']]

files = []
for d in labeled_data_dir:
    files += glob.glob(d + "*.npy")


keys = ['label', 'positions', 'types']

csv_labeling_info = {}
# csv_labeling_info.append(['File names']+keys)

for file in tqdm(files):
    f = file.replace(nb_dir,'')
    data_dict = np.load(file, allow_pickle=True).item()
    csv_labeling_info[f] = {}
    for k in keys:
        if k in data_dict.keys():
            csv_labeling_info[f][k] = data_dict[k]
            del data_dict[k]
        else:
            csv_labeling_info[f][k] = ''
    np.save(file, data_dict) 
    
np.save(nb_dir+'/data/data_info/labels',csv_labeling_info)

#%%

unlabeled_data_dir = [nb_dir + '/data/data_files/class-' + s +'/' for s in ['0', '1', '2', '8', '9']]

files = []
for d in unlabeled_data_dir:
    files += glob.glob(d + "*.npy")

for file in tqdm(files):
    f = file.replace(nb_dir,'')
    data_dict = np.load(file, allow_pickle=True).item()
    if len(data_dict.keys())!=6:
        print(data_dict.keys())            
    
