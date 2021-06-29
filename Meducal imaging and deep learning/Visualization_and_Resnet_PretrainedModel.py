# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:54:09 2021

@author: BERETE Mohamed Loua
"""

#testing the accuracy of our model on the dataset using Resnet Weights

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from path import Path
import glob
from PIL import Image

import seaborn as sns
from fastai.vision.all import *
from fastai.medical.imaging import *

root = 'intracranial_hemorrhagies_dataset/train_jpg/train_jpg'
csv = 'intracranial_hemorrhagies_dataset/metadata/labels.csv'

roots = get_image_files(root)

#number of images/ size of the dataset

print(len(roots))


labels = pd.read_csv(csv)

#
labels.head()


#Test : label.columns
#deleting the columns we won't need
labels = labels.drop(['Unnamed: 0', 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid', 'subdural'], axis = 1)

#different ways to view the files

#Number 1
img =PILImage.create(roots[1000], mode = 'RGB')
print(img)

from random import randint

f, axarr = plt.subplots(3,3)
plt.figure(figsize=(30,30)) 
axarr[0,0].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[0,1].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[0,2].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[1,0].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[1,1].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[1,2].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[2,0].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))
axarr[2,1].imshow(PILImage.create(roots[randint(1,1000)], mode = 'RGB'))


#Number 2



img_2 = ImageDataLoaders.from_df(labels, trn_path, label_delim=';', batch_tfms=tfms, seed = 42)


#Finding the perfect learning rate :
#The cnn_learner function from fastai Library will help you define a Learner using a pretrained model, in this case we will use the Resnet pretrained model
learning_rate = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.5), model_dir = '/kaggle')



#We will use this popular function in order to find the perfect learning rate for our classifier

learn.lr_find()


#Trying with multiple pretrained models

learn = cnn_learner(dls, alexnet, metrics=partial(accuracy_multi, thresh=0.5), model_dir = '/kaggle')
