
#Importing the packages
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input,MaxPool2D,Conv2D,UpSampling2D,Activation,BatchNormalization,Subtract, Concatenate,AveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import datetime
import pandas as pd
import time
# create CNN model
from tensorflow.keras.layers  import Concatenate
from tensorflow.keras.layers  import Add
from tensorflow.keras.layers  import Multiply
from tensorflow.keras.layers  import GlobalAveragePooling2D
from tensorflow.keras.layers  import Input,MaxPool2D,Conv2D,UpSampling2D,Activation,BatchNormalization,Subtract, Conv2DTranspose
import glob
import PIL
from PIL import Image
import random
#import tensorflow_addons as tfa

import re
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
from tensorflow.keras.callbacks import TensorBoard
import imp
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

import tensorflow as tf
import datetime

#Getting the filepaths for train and test data
train_files=['./original/'+filename for filename in os.listdir('./original/')]
test_files=['./Noisy/'+filename for filename in os.listdir('./Noisy/')]

train_files.sort()
test_files.sort()

def manish(filename):
  c =0
  for i in filename:
    '''This function performs adding noise to the image given by Dataset'''
    image_string = tf.io.read_file(i)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_tf = tf.cast(image_decoded, tf.float32)/255
    #image = cv2.resize(cv2.UMat(image),(40,40,3))

    image_tf_4D= tf.expand_dims(image_tf,0)
    image_tf_resized_4D = tf.compat.v1.image.resize_bilinear(
        image_tf_4D,
        (256,256)
    )
    if c==0:
      img = tf.expand_dims(tf.squeeze(image_tf_resized_4D),axis = 0)
    else:
      img = tf.concat([img,tf.expand_dims(tf.squeeze(image_tf_resized_4D),axis =0)], axis = 0)
    c=c+1
    #print(c)
  return img
    #noise_level=np.random.choice(NOISE_LEVELS)
    #noisy_image=image
    #noisy_image=tf.clip_by_value(noisy_image, clip_value_min=0., clip_value_max=1.)

BATCH_SIZE=16

train_dataset = manish(np.array(train_files))
test_dataset = manish(np.array(test_files))

print(train_dataset.shape)
print(test_dataset.shape)

def dct_layer(input_tensor):
    return tf.signal.dct(input_tensor, type = 2)

def ID_Module(input):
    x=Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')(input)
    s=Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu')(x)
    s=Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')(s)
    s=Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')(s)
    s=Concatenate()([s,x])
    
    s=Conv2D(128, (3,3), dilation_rate=1,padding='same',activation='relu')(s)
    t=Conv2D(128, (3,3), dilation_rate=2,padding='same',activation='relu')(s)
    t=Conv2D(128, (3,3), dilation_rate=3,padding='same',activation='relu')(t)
    t=Conv2D(128, (3,3), dilation_rate=4,padding='same',activation='relu')(t)
    t=Concatenate()([s,t])
    
    t=Conv2D(256, (3,3), dilation_rate=1,padding='same',activation='relu')(t)
    s=Conv2D(256, (3,3), dilation_rate=2,padding='same',activation='relu')(t)
    s=Conv2D(256, (3,3), dilation_rate=3,padding='same',activation='relu')(s)
    s=Conv2D(256, (3,3), dilation_rate=4,padding='same',activation='relu')(s)
    s=Concatenate()([s,t])
    
    
    s=Conv2D(128, (3,3), dilation_rate=1,padding='same',activation='relu')(s)
    t=Conv2D(128, (3,3), dilation_rate=2,padding='same',activation='relu')(s)
    t=Conv2D(128, (3,3), dilation_rate=3,padding='same',activation='relu')(t)
    t=Conv2D(128, (3,3), dilation_rate=4,padding='same',activation='relu')(t)
    #t=Concatenate()([s,t])
    
    s=Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')(t)
    t=Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu')(s)
    t=Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')(t) 
    t=Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='sigmoid')(t)  
    z=Multiply()([s,t])
    print(z.shape)
    # Apply DCT to 
    # Apply DCT to the noisy image patches.
    #noise_image_path = 'noise_images/' + str(image_counter) + '.png'
    #image_counter = 1
    #path to training data
    #data= './trainingPatch/img_clean_pats.npy'
    
    #print(data.dtype)
    #tf_load_npy = tf.numpy_function(load_npy, [tf.string], [tf.float32])
    #patches = tf_load_npy(data)
    #raw = tf.io.read_file(data)
    #patches = tf.compat.v1.io.decode_npy(raw)
    #patches = np.load(raw) 
    #patches_tensor= tf.convert_to_tensor(patches, dtype = tf.float64)
    #dst = tf.signal.dct(patches, type = 2, norm = 'ortho')
            
    #dst = np.array([dct(patches) for patch in patches])
    
    #print(noise_dct_data)
    
    dst = dct_layer(input) 
    #s = tf.expand_dims(s,1)
    #print(s.shape)
    #s = tf.expand_dims(s,1)
    #print(s.shape)
    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(dst)
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(dst)
    add = Add()([max_pool,avg_pool])
    #conv_1 = tf.keras.layers.Conv2D(64, 3, dilation_rate=1,padding="same", activation=None)(dst)
    #conv_2 = tf.keras.layers.Conv2D(64, 3, dilation_rate=2,padding="same", activation=None)(conv_1)
    conv_1 = tf.keras.layers.Conv2D(64, 3, dilation_rate=3,padding="same", activation=None)(dst)
    conv_2 = tf.keras.layers.Conv2D(64, 3, dilation_rate=4,padding="same", activation=None)(conv_1)
 
    max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(add)
    avg_pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(max_pool_2)
    conv = tf.keras.layers.Conv2D(64, 3,dilation_rate=3,padding="same", activation=None)(avg_pool_2)
    sigmoid = tf.keras.activations.sigmoid(conv)
    mul = tf.keras.layers.Multiply()([sigmoid, conv_2])
    add=Concatenate()([mul, z])
    return add

    #s=Conv2D(64, (3,3), padding='same',activation='relu')(conc)
    #t=Conv2D(64, (3,3), padding='same',activation='relu')(s)
    #t=Conv2D(64, (3,3), padding='same',activation='relu')(t) 
    #t=Conv2D(64, (3,3), padding='same',activation='sigmoid')(t) 
    #mul=Multiply()([s, t])
    #add=Concatenate()([mul, z])
    #return add
    
def DRSFANET():

    input = Input((None, None, 3),name='input')
    eam_1=ID_Module(input)
    w=Conv2D(3, (3,3),padding='same')(eam_1)
    add_4=Subtract()([input,w])
    print(add_4.shape)
    model=Model(input, add_4)
    return model

tf.keras.backend.clear_session()
tf.random.set_seed(6908)
DRSFANet = DRSFANET()
DRSFANet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanAbsoluteError())
DRSFANet.summary()

#plot_model(ridnet,show_shapes=True,to_file='ridnet.png')

def scheduler(epoch,lr):
  return lr*0.9

# Commented out IPython magic to ensure Python compatibility.
checkpoint_path = "./DRSFANET.h5" # For each epoch creaking a checkpoint
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=0,save_best_only=False) # To save the model if the metric is improved
#! rm -rf "./logs_ridnet/"
# Tensorbaord  # Removing all the files present in the directory
#logdir = os.path.join("logs_ridnet", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Directory for storing the logs that are required for tensorboard
# %tensorboard --logdir $logdir
#tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
train_dataset.shape
train_dataset.shape
callbacks = [cp_callback,lrScheduler]
DRSFANet.fit(test_dataset,train_dataset,batch_size=BATCH_SIZE,shuffle=True,epochs=100,callbacks=callbacks)
