# CorrectSitting
This is a project that will make you sit correctly. It will take advante of machine learning algorithms. 

# Requirements

This work was done with Keras and a Tensorflow-Backend. It does not use any custom Tensorflow-Code, but was not tested with any other backend for Keras. 

## Software:

- Keras (Tensorflow backend)
- Numpy
- OpenCV (Python is fine)
- os
- skimage
- tqdm
- easygui
- time

## Hardware

You may execute and train with a CPU, however utilizing a GPU will be much faster. 
A GPU with at least 4GB must be used. 

## Other

At least 1 Gb of free space on your hard drive. 

# How to 

## TL;DR:
The normal pipeline has to be the following:

1. Start TakeData
2. Start SplitAndAugmentation
3. Start TrainData

## TakeData

This script will activate your webcam and prompt messages telling you what to do. It will save images locally to your hardrive. At first a session is started where you should sit correctly for 15 minutes. After that another session is started where you should sit incorrectly for another 15 minutes (sry). 

## SplitAndAugmentation

This script will create train and validation files, for later traing. It will augment the data provided by TakeData.py and augment them

## TrainData

This script will start the training process 
