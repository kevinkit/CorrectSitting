# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:35:52 2017

@author: kevin
"""

import os 

os.chdir('C:\\Users\kevin\Desktop\MyProjects\CorrectSitting')


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input,Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from os import listdir
from os.path import isfile, join
import numpy as np
# path to the model weights files.

# dimensions of our images.
img_width, img_height = 224, 224


train_len = len([f for f in listdir('train/correct') if \
               isfile(join('train/correct', f))])  + \
            len([f for f in listdir('train/incorrect') if isfile(join('train/incorrect', f))])
val_len = len([f for f in listdir('validation/correct') if \
               isfile(join('validation/correct', f))])  + \
            len([f for f in listdir('validation/incorrect') if isfile(join('validation/incorrect', f))])


train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = train_len
nb_validation_samples = val_len
epochs = 400
batch_size = 1



input_tensor_default = Input(shape=(img_width,img_height,3))
# build the VGG16 network
base_model = applications.VGG16(input_tensor=input_tensor_default,weights='imagenet', include_top=False)
print('Model loaded.')

flat = Flatten(name='flatten')(base_model.output)
#x = base_model.output
x = Dense(1024,activation='relu')(flat)
x = Dense(1024, activation='relu')(x)

predictions = Dense(1, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# set the first 20 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:20]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)