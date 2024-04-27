#importLibraries
import os.path
import tensorflow as tf
from tensorflow.keras.models import load_model
from conf import myConfig as config
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from pathlib import Path
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
import scipy.io
from scipy import ndimage
import hdf5storage
from tifffile import imwrite
from utils import utils_image as util
from PIL import Image
#import skimage.feature
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

#ParsingArguments
parser=argparse.ArgumentParser()
parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Testing_data/BSD68/',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='./Pretrained_models/DRSFANet_Gray.h5.h5',help='pathOfTrainedCNN')
args=parser.parse_args()
#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res

nmodel_PROPOSED=load_model(args.weightsPath,custom_objects={'my_Hfilter': my_Hfilter,'my_Vfilter': my_Vfilter,'custom_loss':custom_loss})
print('Trained Model is loaded')

#createArrayOfTestImages
p=Path(args.dataPath)
listPaths=list(p.glob('./*.png'))
imgTestArray = []
for path in listPaths:
    imgTestArray.append((
    (cv2.imread(str(path),0))))
imgTestArray=np.array(imgTestArray)/255.

noise_level_img = 30 #noise level for noisy image
length =68
sumPSNR=0
sumSSIM=0
psnr_val=np.empty(length)
ssim_val=np.empty(length)
for i in range(0,length):
    np.random.seed(seed=0)  # for reproducibility
    img1=imgTestArray[i]
    f=img1 + np.random.normal(0, noise_level_img/255., img1.shape)
    error=nmodel_PROPOSED.predict(np.expand_dims(f,axis=0))
    predClean=f-np.squeeze(error)
    z=(predClean)
    cv2.imwrite("./results/"+str(i+1)+"_Original.png",255.*img1)
    cv2.imwrite("./results/"+str(i+1)+"_DRSFANet_Gray.png",255.*z)
    cv2.imwrite("./results/"+str(i+1)+"_Noisy.png",255.*f)
    psnr_val[i]=util.calculate_psnr(255.*z,255.*img1)
    ssim_val[i]=util.calculate_ssim(255.*z,255.*img1)
    print('PSNR of image '+str(i+1)+' is ',psnr_val[i])
    print('SSIM of image '+str(i+1)+' is ',ssim_val[i])
    sumPSNR=sumPSNR+psnr_val[i]
    sumSSIM=sumSSIM+ssim_val[i]
avgPSNR=sumPSNR/lenth
avgSSIM=sumSSIM/lenth
print('avgPSNR on dataset = ',avgPSNR)
print('avgSSIM on dataset = ',avgSSIM)
