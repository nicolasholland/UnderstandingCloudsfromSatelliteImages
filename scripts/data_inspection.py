#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:24:28 2019

@author: tzech
"""
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

INPUT_SHAPE = (1400, 2100)
#%%
def rle2mask(rle, input_shape=INPUT_SHAPE):
    """
    Source
    ------
    https://www.kaggle.com/saneryee/understanding-clouds-keras-unet
    """
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def to_image_fname(Image_Label):
    fname = Image_Label.split('_')[0]
    return fname

def to_image_id(Image_Label):
    fname = to_image_fname(Image_Label)
    im_id, ext = os.path.splitext(fname)
    return im_id

def to_label(Image_Label):
    label = Image_Label.split('_')[1]
    return label
#%%

DATADIR = os.path.expanduser('~/data/kaggle/understanding_clouds/')

#%% Reading training labels
fname = os.path.join(DATADIR, 'train.csv')
train = pd.read_csv(fname)
#%% Reading sample submission
sub = pd.read_csv(os.path.join(DATADIR, 'sample_submission.csv'))
#%% Make train and sub more tidy

train['im_fname'] = train['Image_Label'].apply(to_image_fname)
train['im_id'] = train['Image_Label'].apply(to_image_id)
train['label'] = train['Image_Label'].apply(to_label)
sub['im_fname'] = sub['Image_Label'].apply(to_image_fname)
sub['im_id'] = sub['Image_Label'].apply(to_image_id)
sub['label'] = sub['Image_Label'].apply(to_label)
#%% Only for 4 images have masks for all four labels
fil_nan = train['EncodedPixels'].notnull()
train.loc[fil_nan, 'im_id'].value_counts().value_counts()
#%%
n = 4
gridspec_kw = dict(top=0.963, bottom=0.0, left=0.0, right=1.0, hspace=0.0, wspace=0.0)
fig, axs = plt.subplots(4, n, 
                        squeeze=False, figsize=(25, 12),
                        subplot_kw=dict(xticks=[], yticks=[]),
                        gridspec_kw=gridspec_kw)
for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), n)):
    for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):
        ax = axs[i, j]
        im = Image.open(os.path.join(DATADIR, f"train_images/{row['im_fname']}"))
        ax.imshow(im)
        mask_rle = row['EncodedPixels']
        try: # label might not be there!
            mask = rle2mask(mask_rle)
        except:
            mask = np.zeros(INPUT_SHAPE)
        ax.imshow(mask, alpha=0.5, cmap='gray')
        im_label = f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}"
        ax.set_title(im_label, {'fontsize': 6})