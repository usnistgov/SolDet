#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used in soliton metric and process 1D profiles.

Created on Sat May 29 16:11:20 2021
@author: sjguo
"""

import numpy as np
from lmfit import Model, Parameters
from scipy.optimize import curve_fit
from scipy import stats
from scipy.optimize import fmin
from sklearn.preprocessing import PowerTransformer
from tqdm import tqdm
import traceback
import pickle

class MexicanHatMetric():
    def __init__(self):
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        def_name = text[:text.find('=')].strip()
        self.name = def_name
        try:
            self.load()
        except:
            self.pt = PowerTransformer()
            self.cov = np.empty(1)
            self.means = np.empty(1)
            self.dim = 1
            
    def save(self, directory='', selfname=False):
        """save class as self.name.pkl"""
        if selfname:
            filename = directory+self.name+'.pkl'
        else:
            filename = directory+'.pkl'
            
        with open(filename,'wb') as file:
            file.write(pickle.dumps(self.__dict__))

    def load(self, directory=''):
        """try load self.name.pkl"""
        with open(directory,'rb') as file:
            dataPickle = file.read()
        self.__dict__ = pickle.loads(dataPickle)
    
    def fit(self, data, positions=None):
        one_soliton_params = find_soliton(data, positions=positions, return_list=True)
        
        if positions is not None: #flatten nested list
            one_soliton_params = [item for sublist in one_soliton_params for item in sublist]
        self.fit_params = preprocess_mhat_params(one_soliton_params)
        
        self.means, self.cov, self.pt = build_metric(self.fit_params)
        self.dim = len(self.means)
    
    def predict(self, data, positions=None, return_dist=False, flatten=False, return_params=False, use_minimum_as_center=False):
        dim = np.squeeze(data).shape
        soliton_params = find_soliton(data, positions=positions, return_list=True)
        
        if dim == (132, 164):
            if soliton_params == []:
                pred = []
                process_params = []
            else:
                process_params = preprocess_mhat_params(soliton_params, use_minimum_as_center=use_minimum_as_center)
                pred = self.apply_metric(process_params, return_dist=return_dist)
        elif dim[1:] == (132, 164):
            pred = []
            process_params = []
            for params_per_image in soliton_params:
                if params_per_image == []:
                    pred.append([])
                    process_params.append([])
                else:
                    process_params_per_image = preprocess_mhat_params(params_per_image, use_minimum_as_center=use_minimum_as_center)
                    pred.append(self.apply_metric(process_params_per_image, return_dist=return_dist))
                    process_params.append(process_params_per_image)
            if flatten:
                pred = [item for sublist in pred for item in sublist]
                if return_params:
                    process_params = [item for sublist in process_params for item in sublist]
        
        if return_params:
            return pred, process_params
        else:
            return pred
    
    def apply_metric(self, test_params, return_dist=False):
        '''
        Apply metric.
    
        Parameters
        ----------
        test_params : numpy array  
            processed mexican hat params.

        Returns
        -------
        res List of float
            (List of) Metric Score.
    
        '''
        if len(test_params.shape)==1:
            test_params = np.expand_dims(test_params, 0)
            
        test_params = self.pt.transform(test_params)

        res = []
        for x in test_params:
            m_dist_x = np.dot((x-self.means).transpose(),np.linalg.inv(self.cov))
            m_dist_x = np.dot(m_dist_x, (x-self.means))
            if return_dist:
                res.append(np.sqrt(m_dist_x)) # mahalanobis distance
            else:
                res.append(1-stats.chi2.cdf(m_dist_x, self.dim)) # probability
        return res
    
    def assess(self):
        # Plot 2d correlation, Gaussian test
        pass


#0: kink soliton/other
#1: top partial soliton
#2: bottom partial soliton
#3: clockwise solitonic vortex
#4: counterclockwise
#5: tilted
def classify_solitonic_excitation(img_data, position, metric, 
                                  par0_cutoff=np.log(1.57), invpar0_cutoff=-np.log(1.57), 
                                  par4_hardL_cutoff = -0.53, par4_hardG_cutoff=0.75, 
                                  par4_softL_cutoff = -0.41, par4_softG_cutoff = 0.61, 
                                  par1_L_cutoff =-3.00, par1_R_cutoff = 1.14, 
                                  # prod4_cutoff=-0.01, 
                                  use_minimum_as_center=False):
    '''

    Parameters
    ----------
    img_data numpy array of one image
    position : list of one double, marking position of a soliton
    metric : mhat
    par0_cutoff : TYPE, optional
        DESCRIPTION. The default is 1.57.
    invpar0_cutoff : TYPE, optional
        DESCRIPTION. The default is 1.57.
    par4_hardL_cutoff : TYPE, optional
        DESCRIPTION. The default is -0.53.
    par4_hardG_cutoff : TYPE, optional
        DESCRIPTION. The default is 0.75.
    par4_softL_cutoff : TYPE, optional
        DESCRIPTION. The default is -0.41.
    par4_softG_cutoff : TYPE, optional
        DESCRIPTION. The default is 0.61.
    par1_L_cutoff : TYPE, optional
        DESCRIPTION. The default is -3.0.
    par1_R_cutoff : TYPE, optional
        DESCRIPTION. The default is 1.14.

    Returns
    -------
    class_return : integer, class
    0: kink soliton/other
    1: top partial soliton
    2: bottom partial soliton
    3: clockwise solitonic vortex
    4: counterclockwise solitonic vortex
    5: tilted
    '''
    bottom_mask=np.zeros_like(img_data)
    bottom_mask[:int(bottom_mask.shape[0]/2),:]=1
    top_mask=np.zeros_like(img_data)
    top_mask[int(top_mask.shape[0]/2):,:]=1
    
    top_prod = np.multiply(img_data,top_mask)
    bottom_prod = np.multiply(img_data,bottom_mask)
    top_metrics = metric.predict(top_prod, positions=position, return_dist=True, flatten=False, return_params = True, use_minimum_as_center=use_minimum_as_center)
    bottom_metrics = metric.predict(bottom_prod, positions=position, return_dist=True, flatten=False, return_params = True, use_minimum_as_center=use_minimum_as_center)
    #print(top_metrics)
    #print(bottom_metrics)
    
    
    diff0=top_metrics[1][0][0] - bottom_metrics[1][0][0]
    #ratio0=top_metrics[1][0][0]/bottom_metrics[1][0][0]
    #print(ratio0)
    # invratio0=bottom_metrics[1][0][0]/top_metrics[1][0][0]
    #print(invratio0)
    # Add 
    
    diff1=top_metrics[1][0][1] - bottom_metrics[1][0][1] 
    #diff1=top_metrics[1][0][-1]-bottom_metrics[1][0][-1] # chenged to minimum instead of cen.
    #print(diff1)
    diff4=top_metrics[1][0][4]/top_metrics[1][0][2] - bottom_metrics[1][0][4]/bottom_metrics[1][0][2]
    #print(diff4)
    class_return = 0
    pathString = ""
    
    #amplitude check
    # if ratio0<0:
    #     pathString += "AN"
    #     if top_metrics[1][0][0]>0:
    #         pathString += "2"
    #         class_return = 2
    #     else:
    #         pathString += "1"
    #         class_return = 1
    if diff0 > par0_cutoff:
        pathString += "A"
        pathString += "1"
        class_return = 1
    elif diff0 < invpar0_cutoff:
        pathString += "A"
        pathString += "2"
        class_return = 2
    #passed amplitude check
    else:
        pathString += "_"
        #strong assym check
        if diff4 < par4_hardL_cutoff:
            pathString += "b"
            pathString += "3"
            class_return = 3
        elif diff4 > par4_hardG_cutoff:
            pathString += "b"
            pathString += "4"
            class_return = 4
        else:
            pathString += "_"
            #pos check
            if diff1 < par1_L_cutoff:
                #weak assym check
                pathString += "icL"
                if diff4 > par4_softG_cutoff:
                    pathString += "wb"
                    pathString += "4"
                    class_return = 4
                else:
                    pathString += "wbF"
                    pathString += "5"
                    class_return = 5
            elif diff1 > par1_R_cutoff: 
                pathString += "icR"
                #weak assym check
                if diff4 < par4_softL_cutoff:
                    pathString += "wb"
                    pathString += "3"
                    class_return = 3
                else:
                    pathString += "wbF"
                    pathString += "5"
                    class_return = 5 
                    
    # SG: double check for 3/4 (vortices): 
    # if class_return in [3,4]:
    #     prod4 = top_metrics[1][0][4] * bottom_metrics[1][0][4]
    #     if prod4 > prod4_cutoff: 
    #         class_return = 6 # other class
    
    return class_return


def naive_detector(preprocessed_data, half_width=3):
    '''
    Find positions of all local depletions that are wider than 2*half_width+1.

    Parameters
    ----------
    preprocessed_data : 2-D or 4-D numpy array,
        A preprocessed image. Dimension 132 x 164
    half_width : int, optional
        Half width of local depletions. The default is 3.

    Returns
    -------
    positions : list of int
        Centers of local depletions.
    '''
    if len(preprocessed_data.shape)>3:
        naive_pos = []
        for data in tqdm(preprocessed_data):
            naive_pos.append(_naive_detector(data, half_width))
        return naive_pos
    else:
        return _naive_detector(preprocessed_data, half_width)
    
def _naive_detector(preprocessed_data, half_width=3):
    '''
    Find positions of all local depletions that are wider than 2*half_width+1.

    Parameters
    ----------
    preprocessed_data : 2-D numpy array,
        A preprocessed image. Dimension 132 x 164
    half_width : int, optional
        Half width of local depletions. The default is 3.

    Returns
    -------
    positions : list of int
        Centers of local depletions.

    '''
    _, _, roixwithoutbackg, _ = fit_tf_1D_from_image(preprocessed_data)
    positions = []    
    for i in range(half_width+1, len(roixwithoutbackg)-half_width):
        for j in range(half_width):
            if (roixwithoutbackg[i-(j+1)] <= roixwithoutbackg[i-j]) or (roixwithoutbackg[i+(j+1)] <= roixwithoutbackg[i+j]):
                break
            if j == half_width-1:
                positions.append(find_soliton(preprocessed_data, float(i))['cen'])
    i=0
    while i < (len(positions)-1):
        if np.abs(positions[i] - positions[i+1])<3:
            positions.pop(i)
        else:
            i+=1
        
    return positions


def outlier_treatment(datacolumn):
    '''
    Find IQR of a given data distribution.

    Parameters
    ----------
    datacolumn : 1-D list or numpy array of float numbers
        The given data distribution.

    Returns
    -------
    lower_range : Float
        Lower bound of IQR.
    upper_range : Float
        bound of IQR..

    '''
    sorted(datacolumn)
    Q1, Q3 = np.percentile(datacolumn, [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range
    

def preprocess_mhat_params(params, remove_outliers=False, use_minimum_as_center=False):
    '''
    Preporcess the mexican hat parameters: Remove offest, rescale Amp and A.
    If a set of params is given, also do remove outliers with IQR.

    Parameters
    ----------
    params : list or numpy array
        one set of mexican hat fitting param of a list of them.
    remove_outliers: Boolean
        if True, apply IQR outliers removal.
    Returns
    -------
    p : 1D or 2D numpy array
        preprocessed params.

    '''
    
    p = np.array(params)
    if len(p.shape) == 2:
        # p[:, 0] = (p[:, 0] / (1 - ((p[:, 1]-82)/82)**2)**2) # Scale amp with density of cloud
        # p[:, 3] = p[:, 3] * p[:, 2]**2 # Scale A with sigma
        # p[:, 4] = p[:, 4] * p[:, 2] # Scale B with sigma
        if use_minimum_as_center:
            pos = []
            for i,pi in enumerate(p):
                p[i,1] = fmin(m_hat1D, pi[1], tuple(pi), disp=False)[0]
                # p[i,0] = np.log(-m_hat1D(p[i,1], pi[0], pi[1], pi[2], pi[3], pi[4], pi[5]))
            
        p = p[:, :-1] # Remove Offset
        
        if remove_outliers:
            for i in range(p.shape[1]): # Remove outliers
                l, u = outlier_treatment(p[:,i])
                p = p[(p[:,i] < u) & (p[:,i] > l), :]
                
        # for i, pi in enumerate(p):
        #     pi = np.append(pi, pos[i])
        
    elif len(p.shape) == 1:
        # p[0] = (p[0] / (1 - ((p[1]-82)/82)**2)**2) # Scale amp with density of cloud
        # p[3] = p[3] * p[2]**2 # Scale A with sigma
        # p[4] = p[4] * p[2] # Scale B with sigma
        if use_minimum_as_center:
            p1 = fmin(m_hat1D, p[1], tuple(p), disp=False)[0]
            # p[0] = np.log(-m_hat1D(p1, p[0], p[1], p[2], p[3], p[4], p[5]))
            p[1] = p1
        p = p[:-1] # Remove Offset
        p.append(pos) # add minimum as additional information 
    return p
 
   
def build_metric(train_params):
    '''
    Find metric 

    Parameters
    ----------
    train_params : 2D numpy array  
        processed mexican hat params.

    Returns
    -------
    cov : numpy array
        covariance matrix.
    pt : sklearn PowerTransformer object
        the power transformation of .

    '''
    
    #1. power transform
    pt = PowerTransformer()
    pt.fit(train_params)
    train_params_trans = pt.transform(train_params)
    #2. find cov (mean is 0 and std is 1 after power transform)
    cov = np.cov(train_params_trans, rowvar=0)
    means = np.mean(train_params_trans, axis=0)
    
    return means, cov, pt

    
def apply_metric(test_params, sigma, pt=False, mu = None, return_dist=False):
    '''
    Apply metric.

    Parameters
    ----------
    test_params : numpy array  
        processed mexican hat params.
    pt : sklearn PowerTransformer object
        the power transformation of .
    sigma : numpy array
        covariance matrix.
    mu : np array, optional
        list of mean values, not necessary if pt is applied. 
        The default is np.zeros(5).

    Returns
    -------
    res List of float
        (List of) Metric Score.

    '''
    
    if len(test_params.shape)==1:
        test_params = np.expand_dims(test_params, 0)
        
    if pt != False:
        test_params_trans = pt.transform(test_params)
    else:
        test_params_trans = test_params
        
    res = []
    for x in test_params_trans:
        m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(sigma))
        m_dist_x = np.dot(m_dist_x, (x-mu))
        if return_dist:
            res.append(np.sqrt(m_dist_x)) # mahalanobis distance
        else:
            res.append(1-stats.chi2.cdf(m_dist_x, 5)) # probability
    return np.array(res)


def fit_soliton(vec_x, roixwithoutbackg, inti_pos=None, func='gaussian1D', return_fit_curve=False, return_list=False): 
    '''
    Fit soliton from 1D profile with given function and initial position.

    Parameters
    ----------
    vec_x : 1D numpy array
        pixel index list.
    roixwithoutbackg : 1D numpy array
        1D profile.
    inti_pos : int or float or None, optional
        initial postion and amp for 'cen' and 'amp' params. The default is None.
    func : str, optional #TODO: self define func.
        fitting function, can be 'gaussian1D' and 'm_hat1D'. 
        The default is 'gaussian1D'.
    return_fit_curve : boolean, optional
        If True, return the fitting curve. 
        If Flase, return the fitting parameters
        The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if func == 'gaussian1D':
        sol_pos_1Dmodel = Model(gaussian1D)
    elif func == 'm_hat1D':
        sol_pos_1Dmodel = Model(m_hat1D)
    
    search_range = 6
    
    pars_soliton = Parameters()
    if inti_pos is None:
        # old: pars_soliton.add('amp', value = min(roixwithoutbackg), vary = True)
        pars_soliton.add('amp', value = logamp(roixwithoutbackg), vary = True, min=-30, max=10)
        pars_soliton.add('cen', value = np.argmin(roixwithoutbackg), vary = True, min=0.0, max=164.0)
    else:
        # old: pars_soliton.add('amp', value = roixwithoutbackg[int(inti_pos)], vary = True)
        pars_soliton.add('amp', value = logamp(roixwithoutbackg, inti_pos), vary = True, min=-30, max=10)
        pars_soliton.add('cen', value = inti_pos, vary = True, min=inti_pos-search_range/2, max=inti_pos+search_range/2)
    
    if func == 'gaussian1D':
        pars_soliton.add('sigma', value = 3.0, vary = True) ##only for gaussian
    elif func =='m_hat1D':
        # pars_soliton.add('sigma', value = 4.0, vary = True, max=7.0)
        # pars_soliton.add('a', value = 1.2, vary = True, min=0.0, max=4)
        # pars_soliton.add('b', value = 0.0, vary = True, min=-4, max=4)
        pars_soliton.add('sigma', value = 4.0, vary = True)
        pars_soliton.add('a', value = 0.2, vary = True, min=-30, max=10)
        pars_soliton.add('b', value = 0.0, vary = True)
        # pars_soliton.add('b', value = 0.0, vary = True, min=-2, max=2)
        
    pars_soliton.add('offset', value = 0.0, vary = True)
    
    # pars_soliton.add('cen', value = xROI_crop[pos_guess]+xlow, vary = True) #pos_min[0]
    # pars_soliton.add('width', value = 3.0, vary = True)
    # pars_soliton.add('gamma', value = 6.0, vary = True) ##only for lorentz
    # print(roixwithoutbackg, pars_soliton)
    fitSP1D = sol_pos_1Dmodel.fit(roixwithoutbackg, params = pars_soliton, x = vec_x, nan_policy='raise')
    
    if return_fit_curve:
        return fitSP1D.eval()
    else:
        res_soliton = fitSP1D.best_values
        res_soliton['sigma'] = np.abs(res_soliton['sigma'])
        if return_list:
            res_soliton = list(res_soliton.values())
        return res_soliton

def logamp(roixwithoutbackg, pos=None):
    if pos==None:
        pos = np.argmin(roixwithoutbackg)
    
    pos=int(pos)
    if roixwithoutbackg[pos]< -np.exp(-1):
        # return np.log(-min(roixwithoutbackg[pos-3:pos+3]))
        return np.log(-roixwithoutbackg[pos])
    else:
        return -1
        
    

def gaussian1D(x, amp, cen, sigma, offset):
    '''1-d gaussian: gaussian(x, amp, cen, wid, offset)'''
    g1d = amp * np.exp(-(x-cen)**2 /(2* sigma**2)) + offset
     
    #return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2)) +offset
    return g1d


def m_hat1D(x, amp, cen, sigma, a, b, offset):
    '''
    Mexican hat fitting fucntion

    Parameters
    ----------
    x : float
        Function Input.
    amp : float
        Function params: soliton amplitude.
    cen : float
        Function params: soliton center position.
    sigma : float
        Function params: soliton width.
    a : float
        Function params: soliton symmetric shoulder height.
    b : float
        Function params: soliton asymmetric shoulder height.
    offset : float
        Function params: Offset, only used for fitting.

    Returns
    -------
    F(x)

    '''
    return -np.exp(amp) * (1 - ((cen-82)/82)**2)**2 * (1 - np.exp(a)*((x-cen)/sigma)**2 + b*((x-cen)/sigma)) * np.exp(-(x-cen)**2 /(2* sigma**2)) + offset

    # Test:
    # return -np.exp(amp) * (1 - ((cen-82)/82)**2)**2 * (1 - np.exp(a)*((x-cen)/sigma)**2 + 8*np.tanh(b)*((x-cen)/sigma)) * np.exp(-(x-cen)**2 /(2* sigma**2)) + offset
    
    # old:
    # return amp * (1 - ((cen-82)/82)**2)**2 * (1 - a*((x-cen)/sigma)**2 + b*((x-cen)/sigma)) * np.exp(-(x-cen)**2 /(2* sigma**2)) + offset
    
    # Sophie:
    # return amp * (1 - a*(x-cen)**2 + b*(x-cen)) * np.exp(-(x-cen)**2 /(2* sigma**2)) + offset
        
    
def find_soliton(preprocessed_data, positions=None, func='m_hat1D', return_list=False):
    '''
    Return the harizontal soliton position fitting parameters of given single 
    soliton image.
    
    If position is None, return the deepest depletion fitting params.
    
    Parameters
    ----------
    processed_data : 2-D numpy array, with shape (132, 164)
        A preprocessed image.
    positions: list or float or int or None
    func: string
        fitting function, can be 'm_hat1D' or 'gaussian1D'
    Returns
    -------
    soliton_info : dict or list of dict
        harizontal soliton position fitting parameters. See "fit_soliton" 
        function for more information.

    '''
    data = np.squeeze(preprocessed_data)
    if data.shape == (132, 164):
        vec_x, _, roixwithoutbackg, _ = fit_tf_1D_from_image(data)
        
        if positions is None:
            soliton_info = fit_soliton(vec_x, roixwithoutbackg, func=func, return_list=return_list)
        elif type(positions) in [int, float, np.float64]:
            soliton_info = fit_soliton(vec_x, roixwithoutbackg, inti_pos=positions, func=func, return_list=return_list)
        elif type(positions) is list:
            soliton_info = []
            for p in positions:
                soliton_info.append(fit_soliton(vec_x, roixwithoutbackg, inti_pos=p, func=func, return_list=return_list))
    
    elif len(data.shape) == 3 and data.shape[1:] == (132, 164):
        soliton_info = []
        
        if positions is None:
            for d in tqdm(data):
                vec_x, _, roixwithoutbackg, _ = fit_tf_1D_from_image(d)
                soliton_info.append(fit_soliton(vec_x, roixwithoutbackg, func=func, return_list=return_list)) 
        else:
            for i, d in enumerate(tqdm(data)):
                soliton_info_per_image = []
                vec_x, _, roixwithoutbackg, _ = fit_tf_1D_from_image(d)
                for p in positions[i]:
                    soliton_info_per_image.append(fit_soliton(vec_x, roixwithoutbackg, inti_pos=p, func=func, return_list=return_list))
                soliton_info.append(soliton_info_per_image)
    else:
        print('data shape = '+ preprocessed_data.shape)
        return
    return soliton_info
        

def fit_tf_1D_from_image(roi):
    '''Thomas fermi 1D fitting for background removal'''
    vec_x = np.arange(164)
    vec_y = roi.reshape((132,164)).sum(0)
    peaksx, peaksposx = _pickpeak(vec_y, 5)
    
    guess_amp = (np.mean(peaksx) - np.mean(vec_y))
    guess_cen = np.mean(peaksposx)
    guess_sigma = sum(vec_y * (vec_x - guess_cen)**2) / sum(vec_y)
    guess_sigma = np.sqrt(guess_sigma/2)
    guess_offset = np.mean(vec_y)
    res = _fit_tf_1D(vec_x, vec_y, guess_amp, guess_cen, guess_sigma, guess_offset)
    roixwithoutbackg = (vec_y - res["fitfunc"](vec_x))

    return vec_x, vec_y, roixwithoutbackg, res


def _fit_tf_1D(x,y,guess_amp,guess_cen,guess_rx,guess_offset):
    '''Thomas fermi 1D fitting for background removal'''
    guess = np.array([guess_amp, guess_cen, guess_rx,guess_offset])
    first_guess = ThomasFermi1Dsum(x, *guess)    
    popt, pcov = curve_fit(ThomasFermi1Dsum,x,y,p0=guess)
    data_fitted_tf1D =  ThomasFermi1Dsum(x, *popt)
    Amp, cen, rx, offset = popt
    fitfunc = lambda x: ThomasFermi1Dsum(x, *popt)
    return {"amp": Amp, "center": cen, "rx": rx, "offset": offset, 
            "data_fitted_tf1D": data_fitted_tf1D, 
            "fitfunc": fitfunc, "first_guess": first_guess, 
            "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


def ThomasFermi1Dsum(x, amp, cen, rx, offset):
    '''Thomas fermi 1D Model for background removal'''
    b = (1 - ((x-cen)/rx)**2)
    np.maximum(b, 0, b)
    #np.sqrt(b,b)
    tf1d = amp*(b**2) + offset
    return tf1d


def _pickpeak(x, npicks=20):
    #sort array and take index
    idx = np.argsort(-x) # inverse of sort array--take the maximum value first
   
    idx = idx[0:npicks]
    vals = x[idx]
    
    return vals, idx

