#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:12:27 2021

@author: Shanjie Guo, Justyna Zwolak, Amilson Fritsch

This file contains all necessary functions described in Soliton detector paper
(arXiv: 2101.05404). Including preprocessor, pos_regressor

It also conatains the fucntions to train the classification model, 
and the fucntions to generate images in the paper
"""



import os
import glob
import h5py
from tqdm import tqdm
import random
import numpy as np
from lmfit import Model, Parameters
import scipy.ndimage as snd
from sklearn.metrics import roc_curve
from tensorflow.keras.models import load_model

#from soldet.classifier import f1, label_vec_num_conversion, pred_to_labels
from soldet.mhat_metric import _pickpeak


###################################
#### Soliton detector pipeline ####
###################################
'''
Functions that consist the soliton detection pipeline.
'''

def get_raw_data(directory, **data_params):
    '''
    Get .h5 raw data in a folder. These data are unprocessed and should use
    "preprocess" function before training models.
    

    Parameters
    ----------
    directory : string
        Path to the folder of .h5 raw image.
        e.g. "/Data/raw_data/*"
        
    **data_params : dictionary
        Parameters used for loading data.
        'return_files_names': If return additional list of file names. default: False
        'return_metadata': If return metadata (phase imprinting time and 
                           holding time). default: False
        'shuffle': int or boolean. default: False.
                   if True: random seed, if False: No shuffle, 
                   if int: take int as seed.
        'datasize': int. default: N (number of data in directory)
                    number of data returns, if < number of data in directory
                    
    Returns
    -------
    raw_data_list : list of 3-D numpy array
        A list of raw images. Dimension N x (3 x 488 x 648). N is the number of
        files in 'directory'.

    '''
    
    if 'return_files_names' in data_params:
        return_files_names = data_params['return_files_names']
    else:
        return_files_names = False
    
    if 'return_metadata' in data_params:
        return_metadata = data_params['return_metadata']
    else:
        return_metadata = False
        
    if 'shuffle' in data_params:
        if type(data_params['shuffle']) == int:
            if_shuffle = True
            seed = data_params['shuffle']
        elif data_params['shuffle'] == True:
            if_shuffle = True
            seed = None
        elif data_params['shuffle'] == False:
            if_shuffle = False
    else:
        if_shuffle = False
    
    files = glob.glob(os.getcwd()+directory+"/*.h5")
    if if_shuffle:
        random.Random(seed).shuffle(files)
        
    if 'datasize' in data_params:
        if data_params['datasize'] < len(files):
            datasize = data_params['datasize']
        else:
            datasize = len(files)
    else:
        datasize = len(files)
    
    orientation='xy'
    raw_data_list = []
    if return_metadata:
        imprint_time_list = []
        hold_time_list = []
        
    for file in tqdm(files[:datasize]):
        with h5py.File(file, 'r') as h5_file:
        
            atoms = h5_file['images/' + orientation +'/shot1/']['atoms']
            probe = h5_file['images/' + orientation +'/shot2/']['probe']
            background = h5_file['images/' + orientation +'/shot3/']['background']
            img = (atoms, probe, background)
            img = np.float64(img)
            raw_data_list.append(np.array(img))
            if return_metadata:
                imprint_time_list.append(np.round(h5_file['globals'].attrs['imprinting_time'],5))
                hold_time_list.append(np.round(h5_file['globals'].attrs['oscilation_time'],5))
    
    res = [raw_data_list]
    if return_metadata:
        res.append(imprint_time_list) 
        res.append(hold_time_list)
    if return_files_names:
        res.append(files)
    
    if len(res)==1:
        return res[0]
    else:
        return res


def preprocess(raw_data, **prep_params):
    '''
    Preprocess raw data(s). Described in arXiv: 2101.05404

    Parameters
    ----------
    raw_data : 3-D numpy array, or List of 3-D numpy array
        Single (Or a list of) raw data that contains 3 images of atom, probe and background.
        Shape (3, 488, 648)
        
    **prep_params: dict
        'if_mask': Boolean. If apply the elliptical mask. Default: True
        'normalize_range': List of two floats. The min-max of the normalization, or False
                           Default: [-1.0, 3.0]
        'return_cloud_info': Boolean. If return cloud_info, Default: False
        
    Returns
    -------
    processed_data : 4-D numpy array,
        A array of preprocessed images. Dimension N x 132 x 164 x 1
    cloud_info: dict, or list of dict
        TF2D fitting parameters of the cloud. See get_cloud_info function.
        
    '''
    if 'if_mask' in prep_params:
        if_mask = prep_params['if_mask']
    else:
        if_mask = True
        
    if 'normalize_range' in prep_params:
        normalize_range = prep_params['normalize_range']
    else:
        normalize_range = [-1.0, 3.0]
        
    if 'return_cloud_info' in prep_params:
        return_cloud_info = prep_params['return_cloud_info']
    else:
        return_cloud_info = False
    
    if type(raw_data) == np.ndarray:
        if raw_data.shape == (3,488,648):
            combined_data = combine_data_probe_bg(raw_data)
            cloud_info = get_cloud_info(combined_data)
            crop_roatated_data = crop_rotate(combined_data, cloud_info)
            if if_mask:
                masked_data = apply_mask(crop_roatated_data, cloud_info)
                if normalize_range:
                    processed_data = normalize_data(masked_data, normalize_range)
                else:
                    processed_data = masked_data
            else:
                if normalize_range:
                    processed_data = normalize_data(crop_roatated_data, normalize_range)
                else:
                    processed_data = crop_roatated_data
            
            if return_cloud_info:
                return processed_data, cloud_info
            else:
                return processed_data
        
        elif raw_data.shape[1:] == (3,488,648):
            raw_data = list(raw_data) # not the best approach
    
    if type(raw_data) == list:
        processed_data =[]
        cloud_info = []
        for data in tqdm(raw_data):
            combined_data = combine_data_probe_bg(np.array(data))
            cloud_info.append(get_cloud_info(combined_data))
            crop_roatated_data = crop_rotate(combined_data, cloud_info[-1])
            if if_mask:
                masked_data = apply_mask(crop_roatated_data, cloud_info[-1])
                if normalize_range:
                    processed_data.append(normalize_data(masked_data, normalize_range))
                else:
                    processed_data = masked_data
            else:
                if normalize_range:
                    processed_data.append(normalize_data(crop_roatated_data, normalize_range))
                else:
                    processed_data = crop_roatated_data
        
        if return_cloud_info:
            return np.array(processed_data), cloud_info
        else:
            return np.array(processed_data)


def combine_data_probe_bg(raw_data):
    '''
    A part of preprocess: Combine 3 raw images (atom, probe, dark) into single atom image.

    Parameters
    ----------
    raw_data : numpy array, with shape (3, 488, 648)
        Raw data.

    Returns
    -------
    naive_OD : numpy array, with shape (488, 648)
        Optical depth of the image, the pixel values represent atom density.

    '''
    probedark = raw_data[1] - raw_data[2]
    probedark[probedark == 0] = 1e-8
    absorbed_fraction = (raw_data[0] - raw_data[2]) / probedark
    countMax = int(np.nanmax(raw_data[1]))
    darkVar = np.nanvar(raw_data[2])
    ODMAX_meaningful = -np.log(np.sqrt(darkVar + countMax) / countMax)

    absorbed_fraction[absorbed_fraction<=0] = 1e-3
    naive_OD = -np.log(absorbed_fraction)
    naive_OD[naive_OD<-1.0] = -1.0
    naive_OD[naive_OD>(ODMAX_meaningful+1)] = ODMAX_meaningful + 1
    
    return naive_OD


def get_cloud_info(naive_OD, angle=43):
    '''
    A part of preprocess: Fit TF 2D parameters with given atom cloud image.

    Parameters
    ----------
    naive_OD : numpy array, with shape (488, 648)
        Full atom density image.
    angle: int or float.
        The angle between camera and atom cloud elongated direction.
    
    Returns
    -------
    cloud_info : dict
        'amp': amplitude (peak density) of the 2D cloud.
        'cenx': x of cloud center postion.
        'ceny': y of cloud center postion.
        'rx': cloud radius in x direction.
        'ry': cloud radius in y direction.
        'offset': offset value of the fitting.
        'theta': same as the input angle, but in radian.

    '''
    fullimgsize = naive_OD.shape
    ylow = 0
    yhigh = fullimgsize[0]
    xlow = 0
    xhigh = fullimgsize[1]
    
    xROI = np.arange(xlow, xhigh)
    yROI = np.arange(ylow, yhigh)
    x, y = np.meshgrid(xROI, yROI)
    
    x1D_distribution = np.sum(naive_OD, 0)
    y1D_distribution = np.sum(naive_OD, 1)
    
    peaksx, peaksposx = _pickpeak(x1D_distribution,5)
    peaksy, peaksposy = _pickpeak(y1D_distribution,5)
    
    ThomasFermi2Drotmodel = Model(ThomasFermi2Drot)
    
    pars = Parameters()
    pars.add('amp', value = 2.0, vary = True)
    pars.add('cenx', value = np.mean(peaksposx) + xlow, vary = True)
    pars.add('ceny', value = np.mean(peaksposy) + ylow, vary = True)
    pars.add('rx', value = 66, vary = True)
    pars.add('ry', value = 56, vary = True)
    pars.add('offset', value = np.min(naive_OD), vary = True)
    pars.add('theta', value = np.radians(angle), vary = False)
    
    fitTF2D = ThomasFermi2Drotmodel.fit(naive_OD.ravel(), params = pars, xy = (x,y))
    cloud_info = fitTF2D.best_values
    return cloud_info


def crop_rotate(naive_OD, cloud_info):
    '''
    A part of preprocess: Crop and rotate the image to emphasize the atom cloud.

    Parameters
    ----------
    naive_OD : numpy array, with shape (488, 648)
        Optical depth of the image, the pixel values represent atom density.
    cloud_info : dict
        TF2D fitting parameters of the cloud. See get_cloud_info function.

    Returns
    -------
    roi : numpy array, with shape (132, 164)
        Atom cloud density image.

    '''
    x_size = 82
    y_size = 66

    center = np.array([cloud_info['cenx'], cloud_info['ceny']])
    angle = np.degrees(cloud_info['theta'])
    atoms_rot, pt_rot = rotate_img(naive_OD, center, angle)  
    
    xROI_crop = np.arange(pt_rot[0]-x_size,pt_rot[0]+x_size)
    yROI_crop = np.arange(pt_rot[1]-y_size,pt_rot[1]+y_size)
    x,y = np.meshgrid(xROI_crop, yROI_crop)

    roi = atoms_rot[yROI_crop,:]
    roi = roi[:,xROI_crop] 
    return roi


def apply_mask(crop_roatated_data, cloud_info):
    '''
    A part of preprocess: Apply elliptical mask to data over the noise.
    
    Parameters
    ----------
    crop_roatated_data : numpy array, with shape (1, 132, 164)
        Atom cloud density image.
    cloud_info : dict
        TF2D fitting parameters of the cloud. See get_cloud_info function.

    Returns
    -------
    masked_data : numpy array, with shape (1, 132, 164)
        Atom cloud density image with mask.
        
    '''
    #extrace image size
    imgsize = crop_roatated_data.shape
   
    #extract rotation angle
    angle_deg = np.degrees(cloud_info['theta'])
    #extract three points defining the ellipse
    
    x0, y0 = cloud_info['cenx'], cloud_info['ceny']
    xx, yx = point_pos(x0, y0, d=cloud_info['rx'], theta_deg=angle_deg)
    xy, yy = point_pos(x0, y0, d=cloud_info['ry'], theta_deg=angle_deg+90)

    points = [(x0,y0),(xx,yx),(xy,yy)]
    #rotate the ellipse 
    rot_pts = rotate_mask(points, angle=angle_deg)
    #overlay the ellipse on the rotated empty array (array of zeros)
    rot_array = snd.rotate(np.zeros((488,648)), angle_deg, reshape=True) 
    mask_rot = in_ellipse(rot_array, rot_pts)
    
    mask_final = mask_rot[int(rot_pts[0][1])-int(imgsize[0]/2):int(rot_pts[0][1])+int(imgsize[0]/2), 
                      int(rot_pts[0][0])-int(imgsize[1]/2):int(rot_pts[0][0])+int(imgsize[1]/2)]

    return crop_roatated_data * mask_final


def normalize_data(crop_roatated_data, normalize_range = [-1.0, 3.0]):
    '''
    A part of preprocess: normalize the data from normalize_range to [0,1],
    and ADD additional dimension for input to the neural network models.

    Parameters
    ----------
    crop_roatated_data : numpy array, with shape (132, 164)
        Atom cloud density image.
    normalize_range : List of two floats. The min-max of the normalization
                      Default: [-1.0, 3.0]

    Returns
    -------
    normalized_data : numpy array, with shape (132, 164, 1)
        Atom cloud density image.

    '''
    inp = crop_roatated_data.reshape(*crop_roatated_data.shape,1)
    normalized_data = (np.array(inp)-(normalize_range[0]))/(normalize_range[1]-(normalize_range[0]))
    return normalized_data


##########################
#### Helper functions ####
##########################
'''
Helper functions used in soliton detection pipepline. Mainly for process raw data.
'''


def ThomasFermi2Drot(xy,amp,cenx,ceny,rx,ry,offset,theta):
    '''
    Thomas Fermi 2D fitting function.

    '''
    
    x, y = xy
    xx = (x - cenx) * np.cos(theta) + (y - ceny) * np.sin(theta)
    yy = (y - ceny) * np.cos(theta) - (x - cenx) * np.sin(theta)
    
    b = 1 - (xx/rx)**2 - (yy/ry)**2
    b = np.maximum(b, 0)
    tf2d = amp*(b**(3/2)) + offset
    return tf2d.ravel()


def rotate_img(image, point, angle): 
    '''
    Rotating an image (clockwise) and a selected point within the image 
    by a given angle [the angle should be given in degrees]
    '''
    # px, py = point
    im_rot = snd.rotate(image, angle, reshape=True) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2
    angle_rad = np.deg2rad(angle)
    
    R = np.array([[np.cos(angle_rad), np.sin(angle_rad)], 
                  [-np.sin(angle_rad), np.cos(angle_rad)]])
    new_point = R.reshape(2,2)@(point - org_center) + rot_center
    return im_rot, new_point.astype(int)


def point_pos(x0, y0, d, theta_deg):
    '''
    Find coordinates of a point d distance away from (x0,y0) 
    at an angle theta_deg
    '''
    theta_rad = np.deg2rad(theta_deg)
    return int(x0 + d*np.cos(theta_rad)), int(y0 + d*np.sin(theta_rad))


def rotate_mask(points, angle=43): 
    '''
    Rotating (clockwise) a list of points within the image 
    by a given angle [the angle should be given in degrees]
    '''
#     mask = np.zeros((488,648))
#     mask_rot = snd.rotate(mask, angle, reshape=True) 
    org_center = np.array((323.5, 243.5))#(np.array(mask.shape[:2][::-1])-1)/2
    rot_center = np.array((403, 399))#(np.array(mask_rot.shape[:2][::-1])-1)/2
    angle_rad = np.deg2rad(angle)
    
    R = np.array([[np.cos(angle_rad), np.sin(angle_rad)], 
                  [-np.sin(angle_rad), np.cos(angle_rad)]])
    points_rot = [(R.reshape(2,2)@(point-org_center)+rot_center).astype(int) 
                  for point in points]
    
    return points_rot#, mask_rot


def in_ellipse(arr, pts):
    '''
    Checking which point within an array arr lay inside an ellipse defined 
    by points in pts, where pts = [ellipsis_center, vertex, co-vertex]
    '''
    rx = pts[1][0]-pts[0][0] 
    ry = pts[2][1]-pts[0][1]
    
    return np.array([(x-pts[0][0])**2/rx**2+(y-pts[0][1])**2/ry**2 <= 1 
                     for y in range(arr.shape[0]) 
                     for x in range(arr.shape[1])]).reshape(arr.shape)

def find_rawdata_file(labeled_file_name):
    '''
    find raw data h5 file with labeled npy file name. May need modification.

    Parameters
    ----------
    labeled_file_name : str
        labeled npy file name..

    Returns
    -------
    raw_file_name : str
        raw data h5 file.

    '''
    label_ind = labeled_file_name.find('soliton_images/labeled_data') +28
    raw_file_name = labeled_file_name[:label_ind] + labeled_file_name[label_ind+2:label_ind+17] + labeled_file_name[label_ind+1:]
    raw_file_name = raw_file_name.replace('soliton_images/labeled_data','raw_data')
    raw_file_name = raw_file_name.replace('.npy','.h5')
    return raw_file_name
    

###########################
#### Metadata Analysis ####
###########################
'''
Functions used for metadata, e.g. t_hold, phase_imprint.
'''

def plot_phasespace(**plot_params):
    pass


######################
#### Testing Zone ####
######################
    

# Task: Add position to labeled single soliton data
