# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:54:09 2021

@author: BERETE Mohamed Loua
"""


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
%matplotlib inline

#We are going to use pretrained trained convolutional network, and builds on top of them
#we are using the VGG model which is a model that has already been trained on the ImageNet dataset
#Thus this model is able to recognize low level features

pp = pprint.PrettyPrinter(indent=4)
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)

#we are going to run the test on an ilage of The Louvre in Paris
content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image);

#Let's start by implementing the cost function: 
    
def cost_function(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    

    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G 
    a_C_unrolled =  tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled =  tf.reshape(a_G, shape=[m, n_H * n_W, n_C])
    
    # compute the cost with tensorflow 
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C,a_G)))

    
    return J_content

"""Our goal is to be able to change the style of the image we initialized previously 
The style matrix is also called a "Gram matrix."

#This matrix is important because it is able to capture the prevalence of certain features
#in an image, so it's a matrix that can capture the style of an image """


def gram_matrix(A):
    
#A -- matrix of shape (n_C, n_H*n_W)
    
#    GA -- Gram matrix of A, of shape (n_C, n_C)

    
    
    GA = tf.matmul(A,tf.transpose(A))
    
    
    return GA

""" Our goal will be to minimize the distance between the Gram matrix of the "style" image S and the gram matrix of the "generated" image G. 
for this, we will only use one layer
""" 

#Now let's implement this style cost function

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
   
    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    # Computing gram_matrices for both images S and G 
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss 
    J_style_layer = tf.reduce_sum(tf.square(GS-GG))/(4 * n_C**2 * (n_W*n_H)**2)
    
    
    
    return J_style_layer