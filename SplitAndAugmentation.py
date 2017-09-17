# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:37:24 2017


Will augment and split up the data into training, testing and validation


@author: kevin
"""

import os
from os import listdir
from os.path import isfile, join
import cv2
from tqdm import tqdm
import numpy as np
from skimage import color
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
def hueshift(D_img):

    HSV_D = color.rgb2hsv(D_img)
    HSV_D[:,:,0]  = np.clip(HSV_D[:,:,0]*np.random.uniform(),0,1)

    colorf = color.hsv2rgb(HSV_D)
    return np.asarray(255*colorf,dtype=np.uint8);
   
def augmentation(img):
    new_img = np.zeros((480,640,3),dtype=np.uint8)
#    t = np.random.randint(0,2)
#    if t == 0:
    new_img[:,:,np.random.randint(0,3)] = img[:,:,np.random.randint(0,3)] 
    new_img[:,:,np.random.randint(0,3)] = img[:,:,np.random.randint(0,3)]
    new_img[:,:,np.random.randint(0,3)] = img[:,:,np.random.randint(0,3)]
#    elif t == 1:
#        
#        #CHange to rgb
#        new_img[:,:,0] = img[:,:,2]
#        new_img[:,:,1] = img[:,:,1]
#        new_img[:,:,2] = img[:,:,0]
#        
#        new_img = hueshift(img)
    return new_img;



if not os.path.exists('train'):
    os.makedirs('train')

if not os.path.exists('validation'):
    os.makedirs('validation')
    
os.chdir('train')
if not os.path.exists('correct'):
    os.makedirs('correct')
if not os.path.exists('incorrect'):
    os.makedirs('incorrect')
    
os.chdir('..')
os.chdir('validation')
if not os.path.exists('incorrect'):
    os.makedirs('incorrect')
if not os.path.exists('correct'):
    os.makedirs('correct')   

os.chdir('..')
correct = [f for f in listdir('correct') if isfile(join('correct', f))]
incorrect = [f for f in listdir('incorrect') if isfile(join('incorrect', f))]
mean_correct = np.zeros((480,640,3),dtype=np.float32)
mean_incorrect = np.zeros((480,640,3),dtype=np.float32)
for name in tqdm(correct):
    #Open the image
    I = cv2.imread('correct/' + name)
    mean_correct = mean_correct + (I / len(correct))
for name in tqdm(incorrect):
    #Open the image
    I = cv2.imread('incorrect/' + name)
    mean_incorrect = mean_incorrect + (I / len(correct))
    

    
for i in tqdm(range(0,5)):
    for name in correct:
        #Open the image
        I = cv2.imread('correct/' + name)
        sub_mean = np.asarray(I,dtype=np.float32) - mean_correct
        n = augmentation(np.clip(sub_mean,0,255))
        f = cv2.resize(n,(224,224))
        a = np.random.randint(0,10)
        if a < 3:
            cv2.imwrite('validation/correct/' +str(i) + name,f)
        else:
            cv2.imwrite('train/correct/' + str(i) + name,f)
for i in tqdm(range(0,5)):
    for name in incorrect:
        #Open the image
        I = cv2.imread('incorrect/' + name)
        sub_mean = np.asarray(I,dtype=np.float32)  - mean_incorrect
        n = augmentation(np.clip(sub_mean,0,255))
        f = cv2.resize(n,(224,224))
        a = np.random.randint(0,10)
        if a < 3:
            cv2.imwrite('validation/incorrect/' + str(i) + name,f)
        else:
            cv2.imwrite('train/incorrect/' + str(i) + name,f)
    
    
    
    