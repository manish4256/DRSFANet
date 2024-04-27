from re import X
from conf import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import GlobalAveragePooling2D
import numpy as np
import tensorflow as tf
from numpy import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: 
	tf.config.experimental.set_memory_growth(gpu, True)
# custom filter
def my_Hfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[0]], [[2]]],
            [[[-1]], [[0]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')
    
def my_Vfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[-2]], [[-1]]],
            [[[0]], [[0]], [[0]]],
            [[[1]], [[2]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')

# create CNN model
from tensorflow.keras.layers  import Concatenate
from tensorflow.keras.layers  import Add
from tensorflow.keras.layers  import Multiply
from tensorflow.keras.layers  import GlobalAveragePooling2D
from tensorflow.keras.layers  import Input,MaxPool2D,Conv2D,UpSampling2D,Activation,BatchNormalization,Subtract, Conv2DTranspose


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
    conv_1 = tf.keras.layers.Conv2D(128, 3, dilation_rate=1,padding="same", activation=None)(dst)
    conv_2 = tf.keras.layers.Conv2D(128, 3, dilation_rate=2,padding="same", activation=None)(conv_1)
    #conv_1 = tf.keras.layers.Conv2D(64, 3, dilation_rate=3,padding="same", activation=None)(input)
    #conv_2 = tf.keras.layers.Conv2D(64, 3, dilation_rate=4,padding="same", activation=None)(conv_1)
 
    max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(add)
    avg_pool_2 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(max_pool_2)
    conv = tf.keras.layers.Conv2D(128, 3,dilation_rate=3,padding="same", activation=None)(avg_pool_2)
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
    
def DRSFANet():

    input = Input((None, None, 1),name='input')
    eam_1=ID_Module(input)
    w=Conv2D(1, (3,3),padding='same')(eam_1)
    add_4=Subtract()([input,w])
    print(add_4.shape)
    model=Model(input, add_4)
    return model
         
tf.keras.backend.clear_session()
tf.random.set_seed(6908)
DRSFANet = DRSFANet()
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")

def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(0,55)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield (noisyImagesBatch,trueNoiseBatch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(learning_rate=0.001)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    lp=K.sum(diff*diff)/(2*config.batch_size)
    return lp
DRSFANet.compile(loss=custom_loss,optimizer=opt)

# train
DRSFANet.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
DRSFANet.save('./Pretrained_models/DRSFANET_Color.h5')
