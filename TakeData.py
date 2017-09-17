# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:10:19 2017

@author: kevin


Test for generating data
"""

import easygui
from tqdm import tqdm
import cv2
import os
from time import sleep
correct_directory = 'correct'
incorrect_directory = 'incorrect'
easygui.msgbox("Welcome to correct sitting. With this script you are able" + \
               "to train the network according to your own data." + "\n" + \
               "At first you have to sit correctly. Images will be taken from"+\
               "you and saved locally on your hard drive." + "\n" + \
               "After that you have to sit incorrectly.\n", title="Intro")

easygui.msgbox("Please sit correctly, images will be taken after closing" +\
               "this window." + "\n \n" + "Please work as usual, while you" +\
               "sit correctly.\n\n\n" + "This will take 15 min", \
               title="Correct Sitting")

cap = cv2.VideoCapture(0)


if not os.path.exists(correct_directory):
    os.makedirs(correct_directory)
    
if not os.path.exists(incorrect_directory):
    os.makedirs(incorrect_directory)

for i in tqdm(range(0,900)):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite(correct_directory + '/' + str(i)+ '.png',frame);
    
    #Sleep for 1 Sec so that the person may move a little to prevent overfitting
    sleep(1)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

easygui.msgbox("Please sit incorrectly, images will be taken after closing" \
               +"this window.\n\n"  +"Please work as usual, while you" +\
               "sit incorrectly.\n\n\n" + "This will take 15 min", title="Incorrect Sitting")

cap = cv2.VideoCapture(0)
for i in tqdm(range(0,9000)):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite(incorrect_directory + '/' + str(i)+ '.png',frame);
    #Sleep for 1 Sec so that the person may move a little to prevent overfitting
    sleep(1)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()