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
    Parameters
    ----------
    rle : str
    input_shapte : (int, int)
    
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
gridspec_kw = dict(top=0.963, bottom=0.0, left=0.0, right=1.0, 
                   hspace=0.0, wspace=0.0)
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
        im_lab = f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}"
        ax.set_title(im_lab, {'fontsize': 6})
        
#%%
#np.array(im.getdata())
#%% Dice score
def dice(a, b):
    """dice score for binary arrays
    
    Parameters
    ----------
    a : numpy.array (n, m)
        values either 0 or 1
    obs : numpy.array (n, m)
        values either 0 or 1
        
    Examples
    --------
    >>> arr = np.array([[0, 1, 0], [1, 1, 0]])
    >>> dice(arr, arr)
    1.0
    
    >>> comp_arr = np.abs(arr - 1)
    >>> dice(arr, comp_arr)
    0.0
    
    Remarks
    -------
    2 * |A intersect B| / (|A| + |B|)
    """
    intersect = a * b
    score = 2 * intersect.sum() / (a.sum() + b.sum())
    return score

class EmptyMask(Exception):
    pass

def im_id2mask(im_id, label):
    row = train.loc[train['im_id'] == im_id]
    try:
        mask = rle2mask(row.loc[row['label'] == label, 'EncodedPixels'].values[0])
    except AttributeError:
        raise EmptyMask
    return mask

def compute_dice(a, b):
    try:
        rle_a = a['EncodedPixels']
        rle_b = b['EncodedPixels']
        return dice(rle2mask(rle_a), rle2mask(rle_b))
    except AttributeError:
        return 0.
    
def gen_random_samples(data, label, n_samples):
    df = data.loc[train['label'] == label]
    rand_im_ids = np.random.choice(df['im_id'].unique(), n_samples)
#    fil_nan = df['EncodedPixels'].notnull()
    df = df.set_index('im_id').loc[rand_im_ids].reset_index()
    return df
#%% Statistical model
# Distribution of box sizes
# Distribution of box positions
# Distribution of labels per image
np.random.seed(42)

label = 'Fish'
pred_samples = gen_random_samples(train, label, len(sub[sub['label'] == label]))
#%%
obs_samples = gen_random_samples(train, label, 2)
#%%
dice_vals = np.array([compute_dice(pred, obs)
                        for c, obs in obs_samples.iterrows() 
                        for k, pred in pred_samples.iterrows()])
print(dice_vals.mean())
#%%
fig, ax = plt.subplots()
ax.hist(dice_vals, bins=30)
#%% Built submission
for label in sub['label'].unique():
    pred_samples = gen_random_samples(train, label, len(sub[sub['label'] == label]))
    col = 'EncodedPixels'
    sub.loc[sub['label'] == label, col] = pred_samples[col]
sub.drop(['im_fname', 'im_id', 'label'], axis=1).to_csv('random_sample_submission.csv',
                                                        index=False)