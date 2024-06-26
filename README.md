# DRSFANet
This repository contains the Deep Residual Feature Extraction and Spatial-Frequency Attention-based Image Denoiser code.

# Pretrained models
The pre-trained models for image denoising are: https://drive.google.com/drive/folders/1Z0XschyUvJw5t_OJxb6GGmXD5fHCb4In

The testing datasets are: https://drive.google.com/drive/folders/1jPHJVFXPAjbSLdWgxQdZ9uuUVO1ycXDa

# Run Experiments

To test for blind Gray denoising using DRSFANet write:

python Test_gray.py

The resultant images will be stored in 'results/'

To test for blind Color denoising using DRSFANet write:

python Test_colour.py

To test for real denoising using DRSFANet write:

python testing.py

The resultant images will be stored in 'results/'

Image-wise PSNR & SSIM and Average PSNR & Average SSIM for the whole image database are also displayed in the console as output.
