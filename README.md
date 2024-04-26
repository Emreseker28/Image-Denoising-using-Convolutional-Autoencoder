# Image Denoising using Convolutional Autoencoder

This MATLAB code implements a convolutional autoencoder for denoising images using MATLAB's Neural Network Toolbox. The autoencoder is trained on a dataset of noisy images and learns to reconstruct clean images.

## Dataset
The code uses the DigitDataset provided by MATLAB's Neural Network Toolbox. The dataset consists of grayscale images of handwritten digits (0-9). 

## Preprocessing
- The dataset is split into training, validation, and testing sets.
- Noise is added to the images to create a noisy dataset.
- Common preprocessing steps include resizing the images to 32x32 pixels and rescaling the pixel values.

## Model Architecture
The autoencoder consists of:
- Input layer: Image input layer of size 32x32x1.
- Encoding layers: Convolutional and max pooling layers to extract features.
- Decoding layers: Transposed convolutional layers for upsampling and reconstructing the images.
- Output layer: Regression layer to predict the denoised image.

## Training
- The model is trained using the Adam optimizer.
- Training progresses for a maximum of 50 epochs with a mini-batch size equal to the size of the training dataset.
- Validation data is used to monitor the model's performance and prevent overfitting.
- The best model based on validation loss is saved.

Training process:

![image](https://github.com/Emreseker28/Image-Denoising-using-Convolutional-Autoencoder/assets/54375145/817e608c-ce5d-4a4f-9cc9-ded6796ee62e)


## Evaluation
- The trained model is used to denoise images from the testing dataset.
- The denoised images are compared with the original clean images to compute the Peak Signal-to-Noise Ratio (PSNR) to evaluate the denoising performance.

An example output:

![image](https://github.com/Emreseker28/Image-Denoising-using-Convolutional-Autoencoder/assets/54375145/766381b6-e3ed-47c9-81b4-c75ff77b074a)


## Functions
- `addNoise`: Adds salt and pepper noise to images.
- `commonPreprocessing`: Resizes and rescales images to a common size.
- `augmentImages`: Augments images by randomly rotating them.

## Usage
1. Ensure MATLAB with Neural Network Toolbox is installed.
2. Download and prepare the DigitDataset.
3. Modify the paths as needed.
4. Run the script to train and evaluate the model.

