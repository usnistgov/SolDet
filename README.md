# SolDet
SolDet is an object-oriented package for solitonic fearture detection in absorption images of Bose-Einstein condensates, with wider use for cold atom image analysis. 
It is featured with classifier, object detector, and Mexican hat metric methods. 
Technical details are explained in https://arxiv.org/abs/2111.04881.

The dataset used to prepare SolDet is currently being prepared for public release. Until then, it can be requested from [Justyna Zwolak](jpzwolak@nist.gov).
Note that SolDet assumes the following folder structure for the data files:

```
SolDet
|- ...
|- data
    |- data_files
        |- class-0
        |- class-1
        |- ...
        |- class-9
    |- data_info
        |- data_roster.csv
        |- data_roster.npy
|- ...
```

## Installation
(1) (Recommended) Create an Anaconda environment with python 3.7, then activate it
```
conda create -n SDenv python=3.7
conda activate SDenv
```

(2) Clone this repository and change directory to it
```
cd your_directory/SolDet
```
Replace <em>your_directory</em> with your root directory.

(3) You can install SolDet by running:
```
python setup.py build  
python setup.py install
```

## Getting started

All features of SolDet are integrated in the `SolitonDetector` object. To start using SolDet:
```
from soldet import SolitonDetector
sd = SolitonDetector()
```

## Import data

To import labeled dataset, use `load_data` method:
```
sd.load_data(directory='directory_str', data_key={'train':0.9, 'test':0.1}, data_type='labeled') 
```

To import unlabeled dataset, use
```
sd.load_data(directory='directory_str', data_key='new', data_type='unlabeled') 
```

## Train models

To train an ML model, use use `train_ML` method:
```
sd.train_ML(model_key='object_detector', data_key='train', save=save_str)
```
where `model_key` can be `'object_detector'` or `'classifier'`.

To train the physics-based model, use `define_PIE_classifier` method:
```
sd.define_PIE_classifier(data_key='train', save=save_str)
```
to define a phycics-informed exictiation (PIE) classifier. 
Use `train_quality_estimator` method:
```
sd.train_quality_estimator(data_key='train', save=save_str) 
```
to train a quality estimator.

## Predict and label with pre-trained models

To predict new or test data, use `predict` method:
```
pred = sd.predict(data_key='test', model_key=model_key)
```
where `model_key` can be `'classifier'`, `'object_detector'`, `'PIE_classifier'`, or `'quality_estimator'`. 

To filter predicted data with certain criteria, use `pipeline` method:
```
checks ={'classifier_labels':[1,2], 
         'object_detector_positions_length':[1,2,3],
         'PIE_types':[0], 
         'quality_estimates_bounds':[0.3,1]}
refined_data = sd.pipeline(data_key='new', new_data_key='refined', checks=checks)
```



