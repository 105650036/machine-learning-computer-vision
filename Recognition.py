# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:50:26 2019

@author: Administrator
"""

from os import listdir
from os.path import join
import glob
import numpy as np
from keras.preprocessing.image import  img_to_array, load_img
from PIL import Image
import keras as keras
import pandas as pd
import matplotlib.pyplot as plt

images_all=[]
labels_all=[]
labels=["abraham_grampa_simpson",
"apu_nahasapeemapetilon",
"bart_simpson",
"charles_montgomery_burns",
"chief_wiggum",
"comic_book_guy",
"edna_krabappel",
"homer_simpson",
"kent_brockman",
"krusty_the_clown",
"lenny_leonard",
"lisa_simpson",
"marge_simpson",
"mayor_quimby",
"milhouse_van_houten",
"moe_szyslak",
"ned_flanders",
"nelson_muntz",
"principal_skinner",
"sideshow_bob"]
mypath = "F:\class\machine learning\CNN\\image"
# 取得所有檔案與子目錄名稱
files = listdir(mypath)
# 以迴圈處理
for f in files:
  # 產生檔案的絕對路徑
  fullpath = join(mypath, f)
  image=np.load(fullpath)
  if len(images_all)!=0:
       images_all=np.append(images_all,image,axis=0)
  else :    
       images_all=image

mypath = "F:\class\machine learning\CNN\\hot"
# 取得所有檔案與子目錄名稱
files = listdir(mypath)
# 以迴圈處理
for f in files:
  # 產生檔案的絕對路徑
  fullpath = join(mypath, f)
  label=np.load(fullpath)
  if len(labels_all)!=0:
       labels_all=np.append(labels_all,label,axis=0)
  else :    
       labels_all=label       

test=np.load("F:\class\machine learning\CNN\\test\\test_images.npy")
labels_all = keras.utils.to_categorical(labels_all, num_classes=len(labels))
#%% VGG 16 only for classification
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
# Generate model
model = Sequential()
# input: 190x190 images with 3 channels -> (190, 190, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128,128,3),padding='same',name='block1_conv2_1'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',name='block1_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block1_MaxPooling'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_1'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2),name='block2_MaxPooling'))
model.add(Dropout(0.25))
#
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_1'))
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_2'))
#model.add(Conv2D(256, (3, 3), activation='relu',padding='same',name='block3_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block3_MaxPooling'))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_1'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_2'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block4_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block4_MaxPooling'))
#model.add(Dropout(0.25))
#
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_1'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_2'))
#model.add(Conv2D(512, (3, 3), activation='relu',padding='same',name='block5_conv2_3'))
#model.add(MaxPooling2D(pool_size=(2, 2),name='block5_MaxPooling'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu',name='final_output_1'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu',name='final_output_2'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='sigmoid',name='class_output'))
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'
model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
#EStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

history=model.fit(images_all, labels_all, batch_size=64, epochs=50,shuffle=True)
prediction=model.predict_classes(images_all)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()