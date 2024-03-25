# DRSFANet
This repository contains the code for Deep Residual Feature Extraction and Spatial-Frequency Attention based Image Denoiser.

# DRSFANet
This repo contains the KERAS implementation of "Blind Gaussian Deep Denoiser Network using Multi-Scale Pixel Attention"


# Run Experiments

To test for blind Gray denoising using DRSFANet write:

python Test_gray.py

The resultant images will be stored in 'Test_Results/Gray/'

To test for blind Color denoising using DRSFANet write:

python Test_color.py

The resultant images will be stored in 'Test_Results/Color/'

Image wise PSNR & SSIM as well as Average PSNR & Average SSIM for the whole image database is also displayed in the console as output.

# Train DRSFANet gray denoising network

To train the MSPABDN gray denoising network, first download the [BSD400 dataset](https://github.com/smartboy110/denoising-datasets/tree/main/BSD400) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Gray.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the DRSFANet model file using:

python DRSFANet_Gray.py

This will save the 'DRSFANet_Gray.h5' file in the folder 'Pretrained_models/'.


# Train DRSFANet color denoising network

To train the DRSFANet color denoising network, first download the [CBSD432 dataset](https://github.com/Magauiya/Extended_SURE/tree/master/Dataset/CBSD432) and save this dataset inside the main folder of this project. Then generate the training data using:

python Generate_Patches_Color.py

This will save the training patch 'img_clean_pats.npy' in the folder 'trainingPatch/'

Then run the DRSFANet model file using:

python DRSFANet_Color.py

This will save the 'DRSFANet_Color.h5' file in the folder 'Pretrained_models/'.

