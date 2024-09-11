# CNN for Number/Letter Classification

## Overview

This project implements a Convolutional Neural Network (CNN) to classify numbers or letters from a Kaggle dataset. The model is trained to predict the correct class (number or letter) with a high level of accuracy, achieving **98% training accuracy** and **96.7% validation accuracy**.

## Dataset

The dataset used in this project is sourced from Kaggle. It contains images of numbers or letters that are used as input for training and testing the CNN model.

- **Input:** Images of numbers or letters
- **Output:** Class labels corresponding to the number or letter
- **Number of Classes:** Depends on the dataset (e.g., digits 0-9 or letters A-Z)
  
You can download the dataset from [Kaggle](https://www.kaggle.com).

### Data Preprocessing

- All images are resized to a fixed size of `IMG_SIZE` before feeding into the CNN.
- Images are normalized by scaling the pixel values to a range of 0 to 1.

## Model Architecture

The model is built using **Keras** with a **Convolutional Neural Network (CNN)** architecture. The architecture consists of multiple convolutional layers followed by max-pooling and dropout layers to prevent overfitting.

### Model Summary

- **Input Layer:** Takes images of size `IMG_SIZE`
- **Convolutional Layers:** 5 layers with ReLU activation and `3x3` filters
- **MaxPooling Layers:** 2 layers for down-sampling
- **Dropout Layers:** To prevent overfitting, dropout is applied after key layers
- **Fully Connected Layer:** 1 dense layer with `1024` units and SELU activation
- **Output Layer:** A softmax layer to output probabilities for each class

### CNN Model Code

```python
CNN_model = Sequential()
CNN_model.add(Input(shape=IMG_SIZE, batch_size=BATCH_SIZE, name='Input'))
CNN_model.add(Conv2D(3, (3,3), strides=1, activation='relu', padding='same'))
CNN_model.add(Conv2D(128, (3,3), activation='relu'))
CNN_model.add(MaxPool2D((3,3)))
CNN_model.add(Conv2D(256, (3,3), activation='relu'))
CNN_model.add(Dropout(0.2))
CNN_model.add(Conv2D(256, (3,3), strides=2, activation='relu', padding='same'))
CNN_model.add(MaxPool2D((2,2)))
CNN_model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
CNN_model.add(Dropout(0.2))
CNN_model.add(Conv2D(1024, (2,2), activation='relu', padding='same'))
CNN_model.add(MaxPool2D(2,2))
CNN_model.add(Flatten())
CNN_model.add(Dense(1024, activation='selu'))
CNN_model.add(Dense(len(mapping), activation='softmax'))
