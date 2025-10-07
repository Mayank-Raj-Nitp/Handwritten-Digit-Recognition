# Handwritten-Digit-Recognition
# ✍️ MNIST Handwritten Digit Recognition using Artificial Neural Networks (ANN)

This project demonstrates the application of a Deep Learning model (specifically, a simple Multi-Layer Perceptron built with Dense layers) for **image classification**. It focuses on the classification of handwritten digits from the famous MNIST dataset, fulfilling the requirements for **Deep Learning** and **ANN implementation**.

## 🚀 Project Overview

The project loads the MNIST dataset, prepares the image data, and trains a Sequential ANN to classify grayscale images of digits (0-9).

### Key Features Demonstrated:

* **Deep Learning on Image Data:** Applying ANN techniques to a large image dataset ($\text{70,000}$ total images).
* **Data Preparation:** **Normalization** of pixel values (scaling from $\text{0-255}$ to $\text{0-1}$) and **One-Hot Encoding** of the target labels.
* **ANN Architecture:** Uses a `Flatten` layer to handle the $\text{28x28}$ image input, followed by multiple densely connected layers with **Softmax** output for multi-class prediction.

## 🛠️ Technology Stack

* **Language:** Python
* **Libraries:**
    * `TensorFlow / Keras`: For building and training the deep learning model, and accessing the built-in MNIST dataset.
    * `NumPy`: For numerical operations and data manipulation.

## ⚙️ Installation and Setup

###  Prerequisites

Ensure you have a modern version of Python installed.
To run this project, you need Python and the necessary libraries (TensorFlow / Keras,numpy) installed .

3. Execution
Save the project code into a file named mnist_ann_classifier.py.

Run the script from your terminal:
python mnist_ann_classifier.py

The script will automatically download the MNIST data (if not cached), preprocess it, train the ANN for a few epochs, and then display the final test accuracy and an example prediction.

📊 Results
The output includes the model's test accuracy, which shows how effectively the ANN can generalize to correctly classify unseen handwritten digits. This demonstrates a core competency in large-scale classification tasks.


