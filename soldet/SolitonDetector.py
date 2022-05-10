#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:36:36 2021

@author: sjguo
"""


from soldet.dataset import get_data, plot_images
from soldet.mhat_metric import find_soliton, apply_metric, build_metric, preprocess_mhat_params, naive_detector, MexicanHatMetric, classify_solitonic_excitation
from soldet.classifier import create_classifier, label_4_to_x, balance_data_by_flip
from soldet.object_detector import create_obj_detector, labels_to_positions, pos_41labels_conversion, Metz_loss, f1_41, f1_merged, augment_data_by_flip, filter_augment_position
from soldet.pipeline import get_raw_data, preprocess
from soldet.tf_helper import label_vec_num_conversion, pred_to_labels, f1, plot_histroy, cross_validate
import numpy as np
import glob
from tensorflow.keras.models import load_model

class SolitonDetector():
    '''
    SolDet is an object oriented package for solitonic fearture detection in 
    absorption images of Bose Einstein condensate. with wider use for cold 
    atom image analysis. 
    Featured with classifier, object detector and Mexican hat metric methods. 
    Technical details is explained in https://arxiv.org/abs/2111.04881.
    '''
    
    data = {}
    
    # IMAGE_SIZE = (132, 164) # Processed image size used in dark soliton dataset

    def load_models(self, directory):
        '''
        load all directories and keys for data and models.
        
        Parameters
        ----------
        filename : str
            file name for loading.

        '''
        files = glob.glob(directory + "*")
        for f in files:
            if f[-7:]=='PIE.pkl':
                self.define_PIE_classifier(cutoff_only=True)
                self.PIE_mhat = MexicanHatMetric()
                self.PIE_mhat.load(f)
            elif f[-6:]=='QE.pkl':
                self.mhat = MexicanHatMetric()
                self.mhat.load(f)
            elif f[-6:]=='cla.h5':
                self.cla = load_model(f, custom_objects={'f1': f1})
            elif f[-10:]=='obj_det.h5':
                self.obj_det = load_model(f, custom_objects={'f1_41': [f1_41], 'Metz_loss': Metz_loss})
            
                
    def load_data(self, directory, data_key={'train':0.9, 'test':0.1}, data_type='labeled', **data_params):
        '''
        Get preprocessed data in a folder.

        Parameters
        ----------
        directory : str
            Path from Soliton-learning folder to the folder of raw image.
            e.g. "/Data/soliton_images/labeled_data/*/"
        data_key : str or dict
            Define keyword(s) for this data subset.
            If need more then one subset, define a dictionary with keywords as 
            keys and entries as the portion of that subset, also need to define 
            e.g. 'train', or {'train':0.9, 'test':0.1}
        data_type: str
            'labeled': read image data, label, and position.
            'unlabeled': only read image data
        **data_params : dict
            See DESCRIPTION for soldet.get_data or soldet.get_raw_data 
            
        '''
        if data_type == 'labeled':
            data_params['return_labels'] = True
            data_params['return_positions'] = True
            data_params['return_types'] = True
            datatypekeys = ['data', 'labels', 'positions', 'types']
        elif data_type == 'unlabeled':
            data_params['return_labels'] = False
            data_params['return_positions'] = False
            data_params['return_types'] = False
            datatypekeys = ['data']
        else:
            raise ValueError("data_type is not in ['labeled', 'unlabeled']")
            return
        
        if 'cwd' not in data_params:
            data_params['cwd'] = False
            
        if type(data_key)==str:
            data_key = {data_key:1}
            
        # if 'return_SGJZ_labels' in data_params:
        #     return_SGJZ_labels = data_params['return_SGJZ_labels']
        # else:
        #     return_SGJZ_labels = False
    
        # if 'return_SGJZ_flags' in data_params:
        #     return_SGJZ_flags = data_params['return_SGJZ_flags']
        # else:
        #     return_SGJZ_flags = False
        
        # if 'return_metadata' in data_params:
        #     return_metadata = data_params['return_metadata']
        # else:
        #     return_metadata = False
    
        # if 'return_rawdata' in data_params:
        #     return_rawdata = data_params['return_rawdata']
        # else:
        #     return_rawdata = False
        
        # if 'return_cloudinfo' in data_params:
        #     return_cloudinfo = data_params['return_cloudinfo']
        # else:
        #     return_cloudinfo = False
        
        # if 'return_files_names' in data_params:
        #     return_files_names = data_params['return_files_names']
        # else:
        #     return_files_names = False

        # can add more like this if needed
        # if return_files_names:
        #     datatypekeys.append('files_names')
        
        # load data
        data_from_dir = get_data(directory, **data_params)
        
        if data_type != 'labeled':
            data_from_dir = [data_from_dir]
        
        if len(data_from_dir) != len(datatypekeys):
            raise ValueError('len(data_from_dir) != len(datatypekeys)')
            return
        
        # check p:
        num_data = len(data_from_dir[0])
        inds = [0]
        p_total = 0
        for p in data_key.values():
            p_total += p
            if p_total > 1:
                raise ValueError('p_total > 1')
                return
            inds.append(int(num_data*p_total))
        
        for j, d in enumerate(data_key.keys()):
            self.data[d] = {}
            for i, dt in enumerate(datatypekeys):
                self.data[d][dt] = data_from_dir[i][inds[j]:inds[j+1]]
        return
    
    def preview_data(self, data_key, n=9, seed=None, sample='labels', **plot_params): #TODO
        '''
        Plot a sample from dataset.

        Parameters
        ----------
        data: str
            Keyword for target dataset.
        n: int
            Total number of data sample. The default is 9.
        seed : int
            seed for select sample data. The default is None.
        sample: str
            method to select data. The default is 'class'.
                'class': evenly sample in each class
                'random': random sample in whole dataset
        **data_params : dict
            See DESCRIPTION for soldet.plot_images

        Returns
        -------
        Figure.

        '''
        pass
    
    def train_ML(self, model_key='object_detector', data_key='train', save=False):
        '''
        Train tensorflow models.

        Parameters
        ----------
        model_key: str
            ML model to train, can be 'object_detector' or 'classifier'. 
            The default is 'object_detector'.
        data_key: str
            keyword of data used for training. default as 'train' 
        save: str or boolean
            if str, save tranined model to directory using this string 
            if False: not save. Can use trained models use attribute .cla or . obj_det
        '''
        
        if model_key == 'classifier':
            train_data = self.data[data_key]['data']
            train_labels = label_vec_num_conversion(label_4_to_x(self.data[data_key]['labels']))
            # balance data by flips
            balanced_train_data, balanced_train_labels = balance_data_by_flip(train_data, train_labels)
            self.cla = create_classifier() # predefined tensorflow model
            self.cla.fit(balanced_train_data, balanced_train_labels, epochs=50)
            if save:
                self.cla.save(save+'cla.h5')

        elif model_key == 'object_detector':
            train_data = []
            train_positions = []
            for i, data in enumerate(self.data[data_key]['data']):
                if self.data[data_key]['labels'][i]<2: #kink
                    train_data.append(data)
                    train_positions.append(self.data[data_key]['positions'][i])
            # augment data by flips
            augmented_train_data, augmented_positions = augment_data_by_flip(train_data, train_positions)
            augmented_train_positions = pos_41labels_conversion(augmented_positions)
            self.obj_det = create_obj_detector()
            self.obj_det.fit(augmented_train_data, augmented_train_positions, epochs=50)
            if save:
                self.obj_det.save(save+'obj_det.h5')
        
        else: # Todo: add customized model, need a define model method
            pass
        
        return
    
    def train_quality_estimator(self, data_key='train', save=False):
        '''
        Train a quality estimator (Mexican hat metric) with giving data.

        Parameters
        ----------
        data_key: str
            keyword of data used for training. The default as 'train' 
        save: str or boolean
            if str, save tranined model to directory using this string 
            if False: not save. Can use trained models use attribute .mhat
        '''
        kink_data = []
        kink_pos = []
        for i, data in enumerate(self.data[data_key]['data']):
            if self.data[data_key]['labels'][i]==1 and self.data[data_key]['types'][i]==[0]: #kink
                kink_data.append(data)
                kink_pos.append(self.data[data_key]['positions'][i])
                
        self.mhat = MexicanHatMetric()
        self.mhat.fit(kink_data, kink_pos)
        
        if save: 
            self.mhat.save(save+'QE')
    
    def show_correlation(self, data, model): #TODO
        '''
        Show correlation plots of different methods, e.g. if 4 methods: show 6 
        plots.

        Parameters
        ----------
        data : str
            Keyword for target dataset.
        model: list of at least 2 str
            Keyword for target model. Can be a list of 'object_detector', 
            'classifier', 'metric', or 'sorter'.
        '''
        pass
    
    def predict(self, data_key='test', model_key='classifier', position_key='positions'):
        '''
        Return prediction from trained methods.

        Parameters
        ----------
        data_key : str
            Keyword for target dataset. The default as 'test' 
        model_key: str, or list of str
            Keyword for target model. Can be 'object_detector', 'classifier',
            'quality_estimator', or 'PIE_classifier', or list of them.
        position_key:str
            Keyword of positions used. The default as 'positions' (labeled positions)
        Returns
        -------
        Predicted results

        '''
        data = self.data[data_key]['data']
        if model_key == 'classifier':
            res = label_vec_num_conversion(pred_to_labels(self.cla.predict(data)))
            self.data[data_key]['classifier_labels'] = res
        elif model_key == 'object_detector':
            res = pos_41labels_conversion(self.obj_det.predict(data))
            self.data[data_key]['object_detector_positions'] = res
        elif model_key == 'PIE_classifier':
            res = self.apply_PIE_classifier(data_key=data_key, position_key=position_key)
        elif model_key == 'quality_estimator':
            positions = self.data[data_key][position_key]
            res = self.mhat.predict(data, positions)
            self.data[data_key]['quality_estimates'] = res
        else:
            raise ValueError("model_key is not in ['classifier', 'object_detector', 'PIE_classifier', 'quality_estimator']")
            return
        
        return res
        
    
    def define_PIE_classifier(self, data_key='train',
                              par0_cutoff=np.log(1.57), invpar0_cutoff=-np.log(1.57), 
                              par4_hardL_cutoff = -0.53, par4_hardG_cutoff=0.75, 
                              par4_softL_cutoff = -0.41, par4_softG_cutoff = 0.61, 
                              par1_L_cutoff =-3.0, par1_R_cutoff = 1.14, save=False,
                              cutoff_only=False):
        '''
        Define PIE classifier. The default parameters as described in paper.
        '''
        self.par0_cutoff = par0_cutoff
        self.invpar0_cutoff = invpar0_cutoff
        self.par4_hardL_cutoff = par4_hardL_cutoff
        self.par4_hardG_cutoff = par4_hardG_cutoff
        self.par4_softL_cutoff = par4_softL_cutoff
        self.par4_softG_cutoff = par4_softG_cutoff
        self.par1_L_cutoff = par1_L_cutoff
        self.par1_R_cutoff = par1_R_cutoff
        
        if not cutoff_only:
            one_data = []
            one_pos = []
            for i, data in enumerate(self.data[data_key]['data']):
                if self.data[data_key]['labels'][i]==1:
                    one_data.append(data)
                    one_pos.append(self.data[data_key]['positions'][i])
                    
            self.PIE_mhat = MexicanHatMetric()
            self.PIE_mhat.fit(one_data, one_pos)
            
            if save: 
                self.PIE_mhat.save(save+'PIE')
    
    def apply_PIE_classifier(self, data_key='test', position_key='positions'):
        '''
        Return the sub class of a detected solitonic excitation. 

        Parameters
        ----------
        data_key: str
            keyword of data used for training. The default as 'train' 
        position_key:str
            Keyword of positions used. The default as 'positions' (labeled positions)

        Returns
        -------
        types : int (sub class of detected solitonic excitation)
            0: longitutional soliton
            1: top partial soliton
            2: bottom partial soliton
            3: clockwise solitonic vortex
            4: counterclockwise solitonic vortex
            5: tilted

        '''
        data = self.data[data_key]['data']
        positions = self.data[data_key][position_key]
        types = []
        for i, d in enumerate(data):
            if type(positions[i]) is list:
                types_d = []
                for position in positions[i]:
                    t = classify_solitonic_excitation(d, [position], metric=self.PIE_mhat, 
                                                      par0_cutoff=self.par0_cutoff, 
                                                      invpar0_cutoff=self.invpar0_cutoff, 
                                                      par4_hardL_cutoff=self.par4_hardL_cutoff, 
                                                      par4_hardG_cutoff=self.par4_hardG_cutoff, 
                                                      par4_softL_cutoff=self.par4_softL_cutoff, 
                                                      par4_softG_cutoff=self.par4_softG_cutoff, 
                                                      par1_L_cutoff=self.par1_L_cutoff, 
                                                      par1_R_cutoff=self.par1_R_cutoff)
                    types_d.append(t)
            else:
                types_d = positions[i]
            types.append(types_d)
            
        self.data[data_key]['PIE_types'] = types
        return types
    
    # def sort(self, data, position, accept_ranges=[]): #TODO
    #     soliton_type = 'kink'
    #     return soliton_type
    
    # def track(self, data):
    #     '''
    #     Return prediction from trained object detectors and metric.

    #     Parameters
    #     ----------
    #     data : str
    #         Keyword for target dataset.
    #     Returns
    #     -------
    #     soliton_positions, metric_scores.

    #     '''
    #     soliton_positions = [35.4, 87.6]
    #     metric_scores = [0.83, 0.97]
    #     return soliton_positions, metric_scores
    
    def pipeline(self, data_key='new', new_data_key='refined', 
                 checks ={'classifier_labels':[1,2], 'object_detector_positions_length':[1,2,3,4,5,6], 
                          'PIE_types':[0], 'quality_estimates_bounds':[0.75,1]}):
        '''
        Apply all models and find desired data, then build a new dataset.
        

        Parameters
        ----------
        data_key : TYPE, optional
            keyword of data used. The default is 'new'.
        new_data_key : TYPE, optional
            assign key of the filtered new data. The default is 'refined'.
        checks : dict
            list of parameter filtering checks implemented. 
            'classifier_labels': classifier output
            'object_detector_positions_length': number of OD found positive detections
            'PIE_types': list of PIE classifier type.
            'quality_estimates_bounds': lower and upper bound of quality estimate.
            The default is {'classifier_labels':[1,2], 'object_detector_positions_length':[1,2,3,4,5,6],                          'PIE_types':[0], 'quality_estimates_bounds':[0.75,1]}.
            PIE_types':[0], 'quality_estimates_bounds':[0.75,1]}
        Returns
        -------
        New dataset.

        '''
        self.data[new_data_key]={}
        data = self.data[data_key]
        # Find images
        removing_index = []
        n_data = len(data['data'])
        for i, (a, b) in enumerate(zip(data['classifier_labels'], data['object_detector_positions'])):
            if a not in checks['classifier_labels']:
                removing_index.append(i)
            elif len(b) not in checks['object_detector_positions_length']:
                removing_index.append(i)
        
        # Find features
        selecting_pos_index = []
        for i, (a, b) in enumerate(zip(data['PIE_types'], data['quality_estimates'])):
            selecting_pos_i_index = []
            for j, (aa, bb) in enumerate(zip(a, b)):
                if aa in checks['PIE_types'] and bb >= checks['quality_estimates_bounds'][0] and bb <= checks['quality_estimates_bounds'][1]:
                    selecting_pos_i_index.append(j)
            selecting_pos_index.append(selecting_pos_i_index)
            if selecting_pos_i_index == [] and (i not in removing_index):
                removing_index.append(i)
                

        # Make New dataset
        for k in data.keys():
            self.data[new_data_key][k] = []
        
        for i in range(n_data):
            for k in data.keys():
                if i not in removing_index:
                    if k in ['object_detector_positions', 'PIE_types', 'quality_estimates']:
                        self.data[new_data_key][k].append([x for j, x in enumerate(data[k][i]) if j in selecting_pos_index[i]])
                    else:
                        self.data[new_data_key][k].append(data[k][i])
        
        return self.data[new_data_key]
        
        
        
        