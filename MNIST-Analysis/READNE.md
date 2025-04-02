# Overview

This project focuses on analyzing the MNIST dataset, a benchmark dataset of handwritten digits (0-9). The goal is to train classification models to accurately recognize digits from images.

## Dataset

The dataset can be found online and can be downloaded in various ways, it consists of:

- 60,000 training images
- 10,000 test images
- Images are 28x28 grayscale digits (0-9)

## Models used

**Multi-layer Perceptron (MLP)**:
- There are two different types of MLP models in this project, the key difference is the smaller number of neurons in the two hidden layers (50 instead of 120 and 84) in the second model.
-  The input layer for both models have 784 features and the output layer has 10 neurons (one for each class).
-  They both use the *ReLu* activation function to remove negative values and the final layer applies *log softmax* to ensure the output represents a probability distribution over the 10 classes.
-  Both the models use the optimizer *Adam*, the first one with a learning rate of 0.001 and the second one with a learning rate of 0.01. They also use the *Cross-Entropy Loss* function to calculate the loss.

**Convolutional Neural Network (CNN)**:
- There is only one type of CNN model used in this project, it has two convolution (hidden) layers and two linear layers.
- The first convolution layer has an input channel of 1, for greyscale images and applies 10 kernels of size 5x5 to output 10 channels. The second convolution layer takes the 10 outputs from the previous layers as inputs and applies 20 kernels of size 5x5 to output 20 channels.
- The first linear layer has an input size of 320, which is obtained by flattening the output from the second convolution layer, and it outputs 50 neurons, which then are the inputs for the second linear layer and it produces 10 outputs.
- Both of the convolution layers use the *ReLu* activation function and use *Max Pooling* (2x2) to remove negative values and reduce the size of the outputs by half.
- The model uses the optimizer *Adam* with a learning rate of 0.001 and use the *Cross-Entropy Loss* function to calculate the loss.


## Findings

- Both the MLP models produce similar results on the validation set with an accuracy of 97% and a loss of 0.1778.
- The CNN model outperforms the MLP model with an accuracy of 99% and a loss of 0.0341  on the validation set.



