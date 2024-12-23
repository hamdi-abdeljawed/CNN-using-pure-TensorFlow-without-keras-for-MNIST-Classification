# CNN-using-pure-TensorFlow-without-keras-for-MNIST-Classification

This project demonstrates the implementation of a Convolutional Neural Network (CNN) using pure TensorFlow to classify digits from the MNIST dataset.
Project Overview

In this project, we load and preprocess the MNIST dataset, define a CNN model from scratch, train the model, and evaluate its performance. We also include a feature to predict handwritten digits from uploaded images.
Features

    CNN Model Architecture:
        Two convolutional layers with ReLU activation.
        Max-pooling for downsampling.
        Two fully connected layers.
    Training:
        Model trained on the MNIST dataset for digit classification.
        Adam optimizer and cross-entropy loss function are used.
    Prediction:
        Upload and predict handwritten digits using the trained model.

Requirements

    -- TensorFlow 2.17.1 (pre-installed in Google Colab)

    You can verify the TensorFlow version by running:

    **import tensorflow as tf
    print(tf.__version__)**

    -- NumPy for array manipulation.

    -- Matplotlib for visualization of results.

Setup Instructions
1. Load and Prepare MNIST Data

The MNIST dataset is loaded and preprocessed as follows:

    Pixel values are normalized to the range [0, 1].
    Images are reshaped to have the shape (batch_size, height, width, channels).
    Labels are one-hot encoded.

2. Define the CNN Model

The CNN model consists of:

    Conv Layer 1: 32 filters with a 3x3 kernel.
    Conv Layer 2: 64 filters with a 3x3 kernel.
    Fully Connected Layer 1: 128 neurons.
    Fully Connected Layer 2 (Output): 10 neurons (for each digit from 0 to 9).

3. Training

The model is trained for 10 epochs using the Adam optimizer and cross-entropy loss.
4. Predicting Handwritten Digits

After training the model, you can upload an image to classify the digit using the trained CNN model.
How to Run

    Clone or upload this notebook into your Google Colab environment.
    Run the cells in order to:
        Load and preprocess the data.
        Define and initialize the CNN model.
        Train the model.
        Evaluate the model on the test set.
        Upload an image and classify the digit.

Example of Usage

    Once the model is trained, you can upload a handwritten digit image to the Colab environment and use the trained model to predict the digit.
