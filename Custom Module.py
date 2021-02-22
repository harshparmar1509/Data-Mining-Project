#!/usr/bin/env python
# coding: utf-8

# In[61]:


import tensorflow as tf
import cv2
import numpy as np
import os
import PIL
import PIL.Image


# In[70]:


class SingleCropEncoderModel(tf.keras.Model):

    def __init__(self):
        super(SingleCropEncoderModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(64, 64, 3))
        self.conv2 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D((2,2),2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        return x


# In[71]:


class ConvTransposeNet(tf.keras.Model):

    def __init__(self):
        super(ConvTransposeNet, self).__init__()
        self.convt1 = tf.keras.layers.Conv2DTranspose(16, (4,4), padding=1,stride=2, activation='relu', input_shape=(8, 8, 48))
        self.convt2 = tf.keras.layers.Conv2DTranspose(1, (6,6), padding=1,stride=4, activation='sigmoid')
 
    def call(self, inputs):
        x = self.convt1(inputs)
        x = self.convt2(x)
        return x 


# In[75]:


class ConvNet(tf.keras.Model):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3,3), padding=1,stride=1, activation='relu', input_shape=(8, 8, 48))
        self.conv2 = tf.keras.layers.Conv2D(1, (3,3), padding=1,stride=1, activation='softmax')
        self.maxpool = tf.keras.layers.MaxPooling2D((2,2),2)
 
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.conv2(x)
        return x 


# In[80]:


class SeccadeModel(tf.keras.Model):
    def __init__(self, name=None, version=2):
        super().__init__()
        self.sc128 = SingleCropEncoderModel(name)
        self.sc256 = SingleCropEncoderModel(name)
        self.sc512 =  SingleCropEncoderModel(name)
        self.decoder = None
        if version == 1:
            convT_net = ConvTransposeNet()
            self.decoder = convT_net
        else:
            conv_net = ConvNet()
            self.decoder = conv_net
        
        
    def forward(self, x):
        W = 64
        x128 = x[:,:W,:,:]
        x256 = x[:, W : 2*W,:,:]
        x512 = x[:, 2*W : 3*W,:,:]
        x128 = self.sc128.forward(x128)
        x256 = self.sc128.forward(x256)
        x512 = self.sc128.forward(x512)
        #encodings = torch.cat((x128, x256, x512), 1) #concatenate along channel axis
        decoded = self.decoder.forward(encodings)
        return decoded


# In[87]:


from tensorflow.keras.preprocessing import image

image_path = '../triplecroppedimage0.jpg'
image = tf.keras.preprocessing.image.load_img(image_path)
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
input_arr.shape


# In[ ]:




