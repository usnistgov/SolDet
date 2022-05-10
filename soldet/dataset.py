#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used for processed dataset.

Created on Sat May 29 16:10:53 2021
@author: sjguo
"""

import os
import sys
import glob
# import h5py
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from soldet.mhat_metric import fit_tf_1D_from_image, fit_soliton
from pathlib import Path

import matplotlib
# plt.style.use('matplotlibrc')
matplotlib.rc("text", usetex = False)

def get_data(directory, **data_params):
    '''
    Get preprocessed data in a folder. Print the total number of data.

    Parameters
    ----------
    directory : str
        Path from Soliton-learning folder to the folder of raw image.
        e.g. "/Data/soliton_images/labeled_data/*/"
    **data_params : dictionary
        Parameters used for loading data.
        
        'return_labels': If return labels, default: True, Must be False if data is unlabeled.
        
        'Ian_label': If use Ian's label. default: True
        
        'mask': If use masked_data or cloud_data. default: True
        
        'batch': Which label batch should be included. default: [1,2,3,4,5,6,7,8]
        
        'return_files_names': If return additional list of file names. default: False
        
        'return_SGJZ_labels': If return additional SG and JZ labels. default: False
        
        'return_SGJZ_flags': If return additional SG and JZ flags. default: False        
        
        'return_metadata': *developing tool. If return metadata (phase 
                           imprinting time and holding time). default: False
        'shuffle': int or boolean. default: False.
                   if True: random seed, if False: No shuffle, 
                   if int: take int as seed.
        'datasize': int. default: N (number of data in directory)
                    number of data returns, if < number of data in directory
        
    Returns
    -------
    labeled_data_list : 3-D numpy array
        A list of labeled images. Dimension N x 132 x 164. N is the number of
        files in 'directory'.
    labels_list : 1-D numpy array dtype=int
        A list of labels with length N.
        0: No soliton
        1: single soliton
        2-3: Other excitations
    labelsSG_list / labelsJZ_list : : 1-D numpy array dtype=int
        A list of labels with length N, assigned by SG / JZ
    files: List of strings. 
        Directories of each labeled data.
        
    '''
    
    if 'return_labels' in data_params:
        return_labels = data_params['return_labels']
    else:
        return_labels = True
        
    if 'return_positions' in data_params:
        return_positions = data_params['return_positions']
    else:
        return_positions = False
        
    if 'return_types' in data_params:
        return_types = data_params['return_types']
    else:
        return_types = False
    
    # if 'Ian_label' in data_params:
    #     Ian_label = data_params['Ian_label']
    # else:
    #     Ian_label = True
    
    if 'mask' in data_params:
        mask = data_params['mask']
    else: 
        mask = True
        
    # if 'batch' in data_params:
    #     batch = data_params['batch']
    # else:
    #     batch=[1,2,3,4,5,6,7,8]
        
    # if 'return_SGJZ_labels' in data_params:
    #     return_SGJZ_labels = data_params['return_SGJZ_labels']
    # else:
    #     return_SGJZ_labels = False
    
    # if 'return_SGJZ_flags' in data_params:
    #     return_SGJZ_flags = data_params['return_SGJZ_flags']
    # else:
    #     return_SGJZ_flags = False
        
    if 'return_metadata' in data_params:
        return_metadata = data_params['return_metadata']
    else:
        return_metadata = False
    
    # if 'return_rawdata' in data_params:
    #     return_rawdata = data_params['return_rawdata']
    # else:
    #     return_rawdata = False
        
    if 'return_cloudinfo' in data_params:
        return_cloudinfo = data_params['return_cloudinfo']
    else:
        return_cloudinfo = False
    
    if 'return_files_names' in data_params:
        return_files_names = data_params['return_files_names']
    else:
        return_files_names = False
    
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
        if_shuffle = True
        seed = None
    
    if 'cwd' in data_params:
        cwd = data_params['cwd']
    else:
        cwd = False
    
    
    if return_labels:
        if 'labels_directory' in data_params:
            labels_directory = data_params['labels_directory']
        else:
            nb_dir = os.path.split(os.getcwd())[0]
            if nb_dir not in sys.path:
                sys.path.append(nb_dir)
            labels_directory = nb_dir + '/data/data_info/data_roster.npy'
            
        # Start
        if cwd:
            labels_file = os.getcwd()+ labels_directory
        else:
            labels_file = labels_directory
        
        labels_dict = np.load(labels_file, allow_pickle=True).item()
    
    if type(directory)==str:
        if cwd:
            files = glob.glob(os.getcwd()+ directory + "*.npy")
        else:
            files = glob.glob(directory + "*.npy")
            
    elif type(directory)==list:
        files = []
        for d in directory:
            if cwd:
                files += glob.glob(os.getcwd()+ d + "*.npy")
            else:
                files += glob.glob(d + "*.npy")
    
    if if_shuffle:
        random.Random(seed).shuffle(files)
    
    if 'datasize' in data_params:
        if data_params['datasize'] < len(files):
            datasize = data_params['datasize']
        else:
            datasize = len(files)
    else:
        datasize = len(files)
    
    inp = []
    oup = []
    
    if return_positions:
        positions_list = []
    if return_types:
        types_list = []
    
    # if return_SGJZ_labels:
    #     oup1 = []
    #     oup2 = []
        
    # if return_SGJZ_flags:
    #     oup3 = []
    #     oup4 = []
    if return_cloudinfo:
        cloud_info_list = []
    if return_metadata:
        imprint_time_list = []
        hold_time_list = []
    # if return_rawdata:
    #     raw_data_list = []
    #     orientation='xy'
    
    
    for file in tqdm(files[:datasize]):
        data_dict = np.load(Path(file.replace('\\','/')), allow_pickle=True).item()
        
        if mask:
            inp += [data_dict['masked_data'].reshape(132,164,1)]
        else:    
            inp += [data_dict['cloud_data'].reshape(132,164,1)] # generates a list of arrays
            
        if return_labels:
            data_dict.update(labels_dict[file[file.find('SolDet/data')+6:].replace('\\', '/')])
            if 'label_v3' in data_dict:
                oup += [data_dict['label_v3']]
            else:
                print('data ('+file+') is unlabeled, use return_labels=False to get data only.')
                return
        
        # if return_labels and (data_dict['label_batch_num'] in batch):
        #     if Ian_label:
        #         oup += [data_dict['label']] # generates a list of arrays
        #     else:
        #         if 'label_AF' in data_dict:
        #             oup += [data_dict['label_AF']]
        #         else:
        #             oup += [data_dict['label']]
        #     if return_SGJZ_labels:
        #         oup1 += [data_dict['label_SG']]
        #         oup2 += [data_dict['label_JZ']]
        #     if return_SGJZ_flags:
        #         if 'question_flag_SG' in data_dict:
        #             oup3.append(1)
        #         else:
        #             oup3.append(0)
        
        #         if 'question_flag_JZ' in data_dict:
        #             oup4.append(1)
        #         else:
        #             oup4.append(0)
        
        if return_positions:
            if 'excitation_position' in data_dict:
                positions_list.append(data_dict['excitation_position'])
            elif data_dict['label_v3'] == 0:
                positions_list.append([])
            else:
                positions_list.append('')
        
        if return_types:
            if 'excitation_PIE' in data_dict:
                types_list.append(data_dict['excitation_PIE'])
            elif data_dict['label_v3'] == 0:
                types_list.append([])
            else:
                types_list.append('')
            
        if return_cloudinfo:
            cloud_info_list.append(data_dict['fitted_parameters'])
        
        if return_metadata:
            imprint_time_list.append(data_dict['imprint_time'])
            hold_time_list.append(data_dict['hold_time'])
            
        # if return_rawdata: # need raw data
        #     raw_file = find_rawdata_file(file)
        #     with h5py.File(raw_file, 'r') as h5_file:
        #         atoms = h5_file['images/' + orientation +'/shot1/']['atoms']
        #         probe = h5_file['images/' + orientation +'/shot2/']['probe']
        #         background = h5_file['images/' + orientation +'/shot3/']['background']
        #         img = (atoms, probe, background)
        #         img = np.float64(img)
        #         raw_data_list.append(np.array(img))
                    
    labeled_data_list = (np.array(inp) - (-1)) / (3 - (-1)) # converts the list to np.array; min-max normalization
    n_samples = labeled_data_list.shape[0]
    print("Total number of samples :", n_samples)
    
    res = [labeled_data_list]
    
    if return_labels:
        labels_list = np.array(oup) # converts the list to np.array
        res.append(labels_list)
        
    if return_positions:
        res.append(positions_list)
        
    if return_types:
        res.append(types_list)
    
    # if return_SGJZ_labels:
    #     labelsSG_list = np.array(oup1)
    #     labelsJZ_list = np.array(oup2)
    #     res.append(labelsSG_list)
    #     res.append(labelsJZ_list)
    
    # if return_SGJZ_flags:
    #     flagsSG_list = np.array(oup3)
    #     flagsJZ_list = np.array(oup4)
    #     res.append(flagsSG_list)
    #     res.append(flagsJZ_list)
    
    if return_cloudinfo:
        res.append(cloud_info_list)
    
    # if return_rawdata:
    #     res.append(raw_data_list)
    
    if return_metadata:
        res.append(imprint_time_list)
        res.append(hold_time_list)
    
    if return_files_names:
        res.append(files)
    
    if len(res) == 1:
        return res[0]
    else:
        return res




def make_soliton_box(cloud_info, soliton_info):
    '''
    Find a box around soliton

    Parameters
    ----------
    cloud_info : dict
        cloud_info from npy file. must have keys 'rx' and 'ry'
    soliton_info : dict
        soliton_info from fitting. must have keys 'cen' and 'sigma'

    Returns
    -------
    (x, y) : Tuple of Floats
        location of left bottom corner.
    width : Floats
        width of box.
    height : Floats
        height of box.

    '''
    
    image_cen_y = 132 / 2
    image_cen_x = 164 / 2
    
    x = soliton_info['cen'] - (3 * np.abs(soliton_info['sigma']))
    width = 6 * np.abs(soliton_info['sigma'])
    
    if x > image_cen_x:
        dx = x-image_cen_x
        height  = 2 * cloud_info['ry'] * np.sqrt(1 - (dx/cloud_info['rx'])**2)

    elif x + width < image_cen_x:
        dx = image_cen_x - (x + width)
        height  = 2 * cloud_info['ry'] * np.sqrt(1 - (dx/cloud_info['rx'])**2)
    else:
        height = 2 * cloud_info['ry']
    
    y = image_cen_y - (height/2)
    return ((x, y), width, height)


def ellipse_y(soliton_pos):
    if abs(soliton_pos-82)>70:
        return 0
    else:
        return np.sqrt(1-(soliton_pos-82)**2/72**2)*62

def draw_solitons(ax, soliton_positions, **plot_params):
    '''
    Draw soliton on a 2D image or 1D profile .

    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
        Drawing axes.
    soliton_positions : list of float
        position to draw.
    **plot_params : dict
        'dim': int, 1 for profile, 2 for image.
        'color': color for drawing, default red
        'linewidth': linewidthfor drawing, default 1

    Returns
    -------
    None.

    '''
    if 'dims' in plot_params:
        dims = plot_params['dims']
    else:
        dims = 2
    
    if 'color' in plot_params:
        color = plot_params['color']
    else:
        color = 'red'
    
    if 'linewidth' in plot_params:
        linewidth = plot_params['linewidth']
    else:
        linewidth = 1

    if 'add_vertical_space' in plot_params:
        y_add = plot_params['add_vertical_space']
    else:
        y_add = 0
    
    if type(soliton_positions) == tuple:
        if dims == 2:
            rect = Rectangle(*soliton_positions, linewidth=linewidth, 
                             edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        elif dims == 1:
            ax.axvline(x=soliton_positions[0][0]+(soliton_positions[1]/2),
                       linewidth=linewidth, color=color)
            ax.axvspan(soliton_positions[0][0], soliton_positions[0][0]+
                       soliton_positions[1], alpha=0.2, color=color)
    elif type(soliton_positions) in [float, np.float32, np.float64]:
        if dims == 2:
            ax.arrow(soliton_positions, 132+2*y_add, 0, -(y_add+64-ellipse_y(soliton_positions)), head_width=3, head_length=5, fc=color, ec=color)
            ax.arrow(soliton_positions, 0, 0, (y_add+64-ellipse_y(soliton_positions)), head_width=3, head_length=5, fc=color, ec=color)
        elif dims == 1:
            ax.axvline(x=soliton_positions, ls='-.', linewidth=linewidth, color=color)
    elif type(soliton_positions) == list:
        for soliton_position in soliton_positions:
            if dims == 2:
                ax.arrow(soliton_position, 132+2*y_add, 0, -(y_add+64-ellipse_y(soliton_position)), head_width=3, head_length=5, fc=color, ec=color)
                ax.arrow(soliton_position, 0, 0, (y_add+64-ellipse_y(soliton_position)), head_width=3, head_length=5, fc=color, ec=color)
            elif dims == 1:
                ax.axvline(x=soliton_position, ls='-.', linewidth=linewidth, color=color)


def plot_images(data_list, title_list=None, soliton_positions=None, **plot_params): 
    '''
    Plot data in a single long image, with 6 in each row. Can box solitons if
    soliton postion information is provided.

    Parameters
    ----------
    data_list : list of 2-D numpy array
        A list of labeled images. Dimension N x 132 x 164.
    title_list : List of strings or None
        Length = N, shown as titles of images. If None, show sample with no 
        titles. If more than one plot for each image, can seperate title with \n.
    soliton_positions:
        1. List of None, tuple ((x,y), width, height) (Draw box), 
        float (Draw vertical line), list of floats (Draw vertical lines)
        
        2. Dict of the lists discribed above. the keys of them indicates the 
        color shows on the plot.
        
    **plot_params: dictionary
        plot_profile: Boolean. Default False. If add plot of the profile data.
                      Or String: 'm_hat1D', 'gaussian1D'
        plot_fullrange: Boolean. Default False. If add plot of the data in [0, 1] range
        plot_per_row: int. Default: 6
    Returns
    -------
    None.

    '''
    
    
    if 'plot_profile' in plot_params:
        plot_profile = plot_params['plot_profile']
    else:
        plot_profile = False
        
    if 'plot_profile_fit' in plot_params:
        plot_profile_fit = plot_params['plot_profile_fit']
    else:
        plot_profile_fit = 'm_hat1D'

    if soliton_positions is None:
        plot_soliton = False
    elif type(soliton_positions) == dict:
        plot_soliton = 'color'
    else:
        plot_soliton = 'red'
    
    if 'plot_fullrange' in plot_params:
        plot_fullrange = plot_params['plot_fullrange']
    else:
        plot_fullrange = False
        
    if 'plot_per_row' in plot_params:
        plot_per_row = plot_params['plot_per_row']
    else:
        plot_per_row = 6
    
    if 'title_font_size' in plot_params:
        title_font_size = plot_params['title_font_size']
    else:
        title_font_size = 8
    
    if 'return_fig' in plot_params:
        return_fig = plot_params['return_fig']
    else:
        return_fig = False
        
    if 'profile_ylim' in plot_params:
        profile_ylim = plot_params['profile_ylim']
    else:
        profile_ylim = False
        
    if 'figure_zlim' in plot_params:
        figure_zlim = plot_params['figure_zlim']
    else:
        figure_zlim = False
    
    if 'vertical_arrange' in plot_params:
        vertical_arrange = plot_params['vertical_arrange']
    else:
        vertical_arrange = False
        
    if 'print_pdf' in plot_params:
        print_pdf = plot_params['print_pdf']
    else:
        print_pdf = False
        
    if 'add_vertical_space' in plot_params:
        add_vertical_space = plot_params['add_vertical_space']
    else:
        add_vertical_space = False
    
    if 'figure_width' in plot_params:
        figure_width = plot_params['figure_width']
    else:
        figure_width = 7.2
    
    if 'lw' in plot_params:
        lw = plot_params['lw']
    else:
        lw = 1
    
    plot_per_image = 1
    if plot_profile==True:
        plot_per_image += 1
    elif plot_profile=='Two':
        plot_per_image += 2
    if plot_fullrange:
        plot_per_image += 1
    
    
    n_img = len(data_list) * plot_per_image
    n_row = int(np.ceil(n_img / plot_per_row))
    # if vertical_arrange:
    #     n_row += plot_per_image-1
    fig = plt.figure(figsize=(figure_width, (figure_width/7.2)*n_row))
    k=0
    cmap = plt.get_cmap('Greys') # viridis (default), 
        
    for i, roi in enumerate(data_list):
        # Plot original image
        if title_list == None:
            titles = ['']*plot_per_image
        else:
            titles = title_list[i].split('\n')
            for j in range(max(0, plot_per_image-len(titles))):
                titles.append('')
                
        if vertical_arrange:
            plt.subplot(n_row, plot_per_row, (i + 1)+plot_per_row*(plot_per_image-1)*(i//plot_per_row))
            plt.title(titles[k % plot_per_image], fontsize=title_font_size, loc='left')
        else:
            plt.subplot(n_row, plot_per_row, k + 1)
            plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            k+=1
            
        roi_plot = np.array(roi).reshape((132,164))
        if add_vertical_space:
            newrow = np.full((add_vertical_space, 164), roi_plot[0,0])
            roi_plot = np.vstack([newrow, roi_plot, newrow])
        
        if figure_zlim:
            plt.pcolormesh(roi_plot, vmin=figure_zlim[0], vmax=figure_zlim[1], rasterized=True, cmap=cmap)
        else:
            plt.pcolormesh(roi_plot, rasterized=True, cmap=cmap)
            
        if plot_soliton == 'red': 
            ax = plt.gca()
            draw_solitons(ax, soliton_positions[i], add_vertical_space=add_vertical_space)
        elif plot_soliton == 'color':
            ax = plt.gca()
            for color, soliton_position in soliton_positions.items():
                draw_solitons(ax, soliton_position[i], **{'color': color}, add_vertical_space=add_vertical_space)
        
        plt.xticks(())
        plt.yticks(())
        
        # Plot image in [0,1] full range
        if plot_fullrange:
            if vertical_arrange:
                plt.subplot(n_row, plot_per_row, (i + 7)+plot_per_row*(plot_per_image-1)*(i//plot_per_row))
                plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            else:
                plt.subplot(n_row, plot_per_row, k + 1)
                plt.title(titles[k % plot_per_image], fontsize=title_font_size)
                k+=1
            
            plt.pcolormesh(roi_plot,vmin=0, vmax=1, rasterized=True, cmap=cmap)
            if plot_soliton == 'red': 
                ax = plt.gca()
                draw_solitons(ax, soliton_positions[i])
            elif plot_soliton == 'color':
                ax = plt.gca()
                for color, soliton_position in soliton_positions.items():
                    draw_solitons(ax, soliton_position[i], **{'color': color}, add_vertical_space=add_vertical_space)
            plt.xticks(())
            plt.yticks(())
            
            
        # Plot 1-D profile of the image
        if plot_profile==True:
            vec_x, vec_y, roixwithoutbackg, res = fit_tf_1D_from_image(roi)
            
            if vertical_arrange:
                plt.subplot(n_row, plot_per_row, (i + 1+6*(plot_per_image-1))+plot_per_row*(plot_per_image-1)*(i//plot_per_row))
                # plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            else:
                plt.subplot(n_row, plot_per_row, k + 1)
                plt.title(titles[k % plot_per_image], fontsize=title_font_size)
                k+=1
            
            plt.plot(roixwithoutbackg, "-b", label="", linewidth=lw)
            plt.plot(vec_y - res["offset"], "-g", label="", linewidth=lw)
            plt.plot(res["fitfunc"](vec_x) - res["offset"], '-k', linewidth=lw)
            plt.plot(vec_x,np.repeat(0, len(roixwithoutbackg)),'--k', linewidth=lw)
            if profile_ylim:
                plt.ylim(profile_ylim)
                if vertical_arrange and i%plot_per_row!=0:
                    plt.yticks(())
            
            if plot_soliton == 'red': 
                ax = plt.gca()
                draw_solitons(ax, soliton_positions[i], **{'dims':1})
                if type(plot_profile_fit) is str:
                    if type(soliton_positions[i]) is not list:
                        soliton_positions[i] = [soliton_positions[i]]
                    for p in soliton_positions[i]:
                        plt.plot(fit_soliton(vec_x, roixwithoutbackg, inti_pos=p, func=plot_profile_fit, return_fit_curve=True),"-",color='purple', label="", linewidth=lw)
            
            elif plot_soliton == 'color':
                ax = plt.gca()
                for color, soliton_position in soliton_positions.items():
                    draw_solitons(ax, soliton_position[i], 
                                  **{'dims':1, 'color': color})
                    
            plt.xticks(())
            if vertical_arrange and i > (len(data_list)-plot_per_row-1):
                plt.xticks((0, 164))
            plt.xlim([0, 164])
            
        elif plot_profile=='Two':
        
            bottom_mask=np.zeros_like(roi)
            bottom_mask[:int(bottom_mask.shape[0]/2),:]=1
            top_mask=np.zeros_like(roi)
            top_mask[int(top_mask.shape[0]/2):,:]=1
            
            top_prod = np.multiply(roi,top_mask)
            bottom_prod = np.multiply(roi,bottom_mask)
            
            vec_x, vec_y, roixwithoutbackg, res = fit_tf_1D_from_image(top_prod)
            
            # if vertical_arrange:
            #     plt.subplot(n_row, plot_per_row, (i + 1+6*(plot_per_image-1))+plot_per_row*(plot_per_image-1)*(i//plot_per_row))
            #     # plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            # else:
                
            plt.subplot(n_row, plot_per_row, k + 1)
            plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            k+=1
            
            plt.plot(roixwithoutbackg, "-b", label="", linewidth=lw)
            plt.plot(vec_y - res["offset"], "-g", label="", linewidth=lw)
            plt.plot(res["fitfunc"](vec_x) - res["offset"], '-k', linewidth=lw)
            plt.plot(vec_x,np.repeat(0, len(roixwithoutbackg)),'--k', linewidth=lw)
            if profile_ylim:
                plt.ylim(profile_ylim)
                if vertical_arrange and i%plot_per_row!=0:
                    plt.yticks(())
            
            if plot_soliton == 'red': 
                ax = plt.gca()
                draw_solitons(ax, soliton_positions[i], **{'dims':1})
                if type(plot_profile_fit) is str:
                    if type(soliton_positions[i]) is not list:
                        soliton_positions[i] = [soliton_positions[i]]
                    for p in soliton_positions[i]:
                        plt.plot(fit_soliton(vec_x, roixwithoutbackg, inti_pos=p, func=plot_profile_fit, return_fit_curve=True),"-",color='purple', label="", linewidth=lw)
            
            elif plot_soliton == 'color':
                ax = plt.gca()
                for color, soliton_position in soliton_positions.items():
                    draw_solitons(ax, soliton_position[i], 
                                  **{'dims':1, 'color': color})
                    
            plt.xticks(())
            # if vertical_arrange and i > (len(data_list)-plot_per_row-1):
            #     plt.xticks((0, 164))
            plt.xlim([0, 164])
            
            vec_x, vec_y, roixwithoutbackg, res = fit_tf_1D_from_image(bottom_prod)
            
            # if vertical_arrange:
            #     plt.subplot(n_row, plot_per_row, (i + 1+6*(plot_per_image-1))+plot_per_row*(plot_per_image-1)*(i//plot_per_row))
            #     # plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            # else:
                
            plt.subplot(n_row, plot_per_row, k + 1)
            plt.title(titles[k % plot_per_image], fontsize=title_font_size)
            k+=1
            
            plt.plot(roixwithoutbackg, "-b", label="", linewidth=lw)
            plt.plot(vec_y - res["offset"], "-g", label="", linewidth=lw)
            plt.plot(res["fitfunc"](vec_x) - res["offset"], '-k', linewidth=lw)
            plt.plot(vec_x,np.repeat(0, len(roixwithoutbackg)),'--k', linewidth=lw)
            if profile_ylim:
                plt.ylim(profile_ylim)
                if vertical_arrange and i%plot_per_row!=0:
                    plt.yticks(())
            
            if plot_soliton == 'red': 
                ax = plt.gca()
                draw_solitons(ax, soliton_positions[i], **{'dims':1})
                if type(plot_profile_fit) is str:
                    if type(soliton_positions[i]) is not list:
                        soliton_positions[i] = [soliton_positions[i]]
                    for p in soliton_positions[i]:
                        plt.plot(fit_soliton(vec_x, roixwithoutbackg, inti_pos=p, func=plot_profile_fit, return_fit_curve=True),"-",color='purple', label="", linewidth=lw)
            
            elif plot_soliton == 'color':
                ax = plt.gca()
                for color, soliton_position in soliton_positions.items():
                    draw_solitons(ax, soliton_position[i], 
                                  **{'dims':1, 'color': color})
                    
            plt.xticks(())
            # if vertical_arrange and i > (len(data_list)-plot_per_row-1):
            #     plt.xticks((0, 164))
            plt.xlim([0, 164])
            
            
    # fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    if print_pdf:
        plt.savefig(print_pdf+'.pdf', bbox_inches='tight') 
    
    if return_fig:
        return fig
    else:
        plt.show()
    