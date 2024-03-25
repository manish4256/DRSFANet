import tensorflow as tf
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import shutil
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import os
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.layers import Layer
#from keras import initializers, regularizers
#from BRDNET import BatchRenormalization
        
# Load the model
model = tf.keras.models.load_model("./DRSFANET.h5")

# Define the paths to the noisy and original images
noisy_image_dir = "./Noisy"
original_image_dir = "./original/"

# Define the path to the output directory for the denoised images
output_dir = "./results/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the list of noisy image filenames
noisy_image_filenames = sorted(os.listdir(noisy_image_dir))
# Loop over each noisy image
psnr_sum = 0.0
ssim_sum = 0.0
# Loop over each noisy image
for filename in noisy_image_filenames:
    # Load the noisy image
    noisy_image_path = os.path.join(noisy_image_dir, filename)
    noisy_img = img_as_float(io.imread(noisy_image_path))

    # Add a batch dimension to the image
    noisy_img = np.expand_dims(noisy_img, axis=0)

    # Denoise the image and calculate PSNR and SSIM
    denoised_img = model.predict(noisy_img)[0]

    # Load the original image
    original_image_path = os.path.join(original_image_dir, filename)
    original_img = img_as_float(io.imread(original_image_path))

    # Calculate PSNR and SSIM
    psnr_val = psnr(original_img, denoised_img)
    ssim_val = ssim(original_img, denoised_img, multichannel=True)
    
    psnr_sum += psnr_val
    ssim_sum += ssim_val
    # Save the denoised image to the output directory
    output_filename = os.path.join(output_dir, filename)
    io.imsave(output_filename, denoised_img)

    # Print the PSNR and SSIM values
    print(f'{filename} PSNR: {psnr_val:.4f}')
    print(f'{filename} SSIM: {ssim_val:.4f}')

    # Append the results to a CSV file
    results_df = pd.DataFrame({
        'filename': [filename],
        'psnr': [psnr_val],
        'ssim': [ssim_val]
    })
    results_csv = os.path.join(output_dir, 'results.csv')
    if os.path.exists(results_csv):
        results_df.to_csv(results_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv, index=False)


# Calculate the average PSNR and SSIM values
num_images = len(noisy_image_filenames)
avg_psnr = psnr_sum / num_images
avg_ssim = ssim_sum / num_images
print(f'Average PSNR: {avg_psnr:.4f}')
print(f'Average SSIM: {avg_ssim:.4f}')
