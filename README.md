# CNNmodel
# Deep Learning Project: Image Classification using CNN and Transfer Learning

## Overview

This project focuses on building and comparing two image classification models using the **CIFAR-10 dataset**:
1. A **Custom Convolutional Neural Network (CNN)** built from scratch.
2. A **Transfer Learning approach** using pre-trained models (VGG16, ResNet50, and MobileNetV2).

The objective is to evaluate the performance of custom and transfer learning architectures and understand the trade-offs involved.

---

## Dataset

### CIFAR-10
- 60,000 color images (32x32 pixels)
- 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Split into: 
  - **Training set**: 50,000 images
  - **Test set**: 10,000 images

---

## Tools & Technologies

- **Language**: Python 3
- **Frameworks**: TensorFlow / Keras
- **Libraries**:
  - NumPy, Matplotlib, Seaborn
  - Scikit-learn (for evaluation metrics)
  - TensorFlow Datasets (for CIFAR-10)
  - OpenCV (optional image processing)

---

## Project Structure

- CNNmodel.ipynb          # Main notebook with CNN and Transfer Learning
- README.md               # This file


## Project Workflow
1. Data Preprocessing (Normalization and label encoding, Visual inspection of class distributions, Resizing images for pre-trained model input compatibility)
2. Custom CNN (Multiple Conv2D + MaxPooling layers, Dense layers with dropout regularization, Compiled with Adam optimizer and categorical_crossentropy)
3. Transfer Learning (Used pre-trained models (VGG16, ResNet50, MobileNetV2) with frozen convolutional base, Added custom dense classification head, Trained with same dataset and preprocessing)
4. Evaluation (Accuracy, Precision, Recall, and F1-score, Confusion matrix for class-wise performance, Comparison of validation accuracy and loss)
