# Facial Attribute Classification using CNN + RNN with Attention

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-%23FFD21F.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)
[![Datasets](https://img.shields.io/badge/Datasets-%23F4B400.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

This project presents a deep learning-based approach to facial attribute classification, specifically addressing the task of binary classification for the "Male" attribute within the CelebA (Large-scale CelebFaces Attributes) dataset. The core innovation of this work lies in the implementation of a hybrid neural network architecture that synergistically combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), further enhanced by a multi-head attention mechanism.

The primary objective is to investigate and demonstrate the efficacy of this combined CNN-RNN-Attention model in extracting and leveraging salient features from facial images for accurate attribute prediction. CNNs are employed to capture the essential spatial hierarchies inherent in visual data, while RNNs are utilized to model potential sequential dependencies among these extracted spatial features. The integration of a multi-head attention mechanism allows the model to selectively focus on the most pertinent image regions, thereby improving discriminative power.

This repository serves as a valuable educational resource, providing a clear and reproducible example of:

-   A complete facial attribute classification pipeline, from data loading to model evaluation.
-   The use of a pretrained CNN (ResNet-18) as a powerful feature extractor.
-   The application of a bidirectional Long Short-Term Memory (LSTM) network to process spatial feature sequences.
-   The implementation of a multi-head attention mechanism to dynamically weight feature importance.
-   The utilization of the Hugging Face `datasets` library for streamlined dataset management and preprocessing.
-   The implementation of model construction, training, and evaluation using PyTorch.
-   Adherence to best practices in data handling, training procedures, and performance monitoring.

It is important to note that the training script provided is configured for demonstration purposes within resource-constrained environments, such as Google Colab. Consequently, training parameters (e.g., number of epochs, dataset subset size) are intentionally limited. While this may affect overall model performance, the repository's primary value lies in its clear illustration of the model architecture and implementation.

## Goal

The fundamental goal of this project is to implement and evaluate a hybrid CNN-RNN architecture, augmented with a multi-head attention mechanism, for the task of facial attribute classification. This overarching goal is achieved through the following key objectives:

1.  **Dataset Acquisition and Preparation:** To acquire and preprocess the CelebA dataset using the Hugging Face `datasets` library. This includes partitioning the dataset into training and validation subsets while employing stratified sampling to preserve class distribution.

2.  **Spatial Feature Extraction:** To leverage a pretrained ResNet-18 model for the extraction of robust spatial features from facial images. The utilization of a pretrained model leverages transfer learning, capitalizing on representations learned from a large-scale image corpus (ImageNet) to enhance performance and accelerate training.

3.  **Sequential Feature Modeling:** To employ a bidirectional Long Short-Term Memory (LSTM) network to model potential sequential relationships among the extracted spatial features. This enables the model to capture contextual dependencies between different facial regions.

4.  **Attention-Based Feature Weighting:** To implement a multi-head attention mechanism that allows the model to adaptively weight the importance of different spatial features. This mechanism enables the model to focus on the most salient image regions for the target attribute, enhancing discriminative capacity.

5.  **Model Training and Evaluation:** To train the composite CNN-RNN-Attention model using an appropriate loss function and optimization algorithm. Furthermore, to evaluate the model's performance on a held-out validation set to assess its generalization ability and prevent overfitting.

6.  **Code Clarity and Reproducibility:** To provide a well-structured, clearly documented, and reproducible implementation of the proposed methodology. This facilitates understanding, replication, and potential extension of this work by other researchers and practitioners.

By pursuing these objectives, this project aims to offer a practical demonstration of a sophisticated deep learning architecture for a relevant computer vision task, while also serving as a valuable educational resource for those interested in the intersection of CNNs, RNNs, and attention mechanisms in image analysis.
