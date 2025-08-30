<div align="center">
  <h1>🐶 DOG BREED CLASSIFICATION</h1>
  <p><i>Deep Learning for Accurate Multi-Class Dog Breed Identification</i></p>
</div>

<br>

<div align="center">
  <a href="https://github.com/brej-29/Logicmojo-AIML-Assignments-DogBreedClassificationTensorFlow">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/brej-29/Logicmojo-AIML-Assignments-DogBreedClassificationTensorFlow">
  </a>
  <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Notebook-Jupyter-orange">
  <img alt="Python Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24+-blueviolet">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.0+-teal">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange">
</div>

<div align="center">
  <br>
  <b>Built with the tools and technologies:</b>
  <br>
  <br>
  <code>Python</code> | <code>TensorFlow 2.x</code> | <code>TensorFlow Hub</code> | <code>Keras</code> | <code>Pandas</code> | <code>NumPy</code> | <code>Matplotlib</code> | <code>scikit-learn</code> | <code>Jupyter Notebook</code>
</div>

---

## **Table of Contents**
* [Overview](#overview)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Data Exploration & Preprocessing](#data-exploration--preprocessing)
* [Modeling & Evaluation](#modeling--evaluation)
* [Model Deployment](#model-deployment)
* [License](#license)
* [Contact](#contact)

---

## **Overview**

This project presents an end-to-end deep learning pipeline for multi-class dog breed classification, leveraging transfer learning with TensorFlow and TensorFlow Hub. The notebook demonstrates:
* Data loading, inspection, and preprocessing for image and label data
* Exploratory data analysis and visualization of class distributions
* One-hot encoding of 120 dog breed labels
* Efficient image preprocessing and batching using TensorFlow's `tf.data` API
* Model construction using a pre-trained MobileNetV2 as a feature extractor
* Training with callbacks (TensorBoard, EarlyStopping) and validation monitoring
* Model evaluation, prediction visualization, and interpretability
* Saving, loading, and deploying trained models
* Preparing Kaggle-compatible CSV submissions and making predictions on custom images

<br>

### **Project Highlights**

- **Dataset:** Kaggle Dog Breed Identification (10,222 training images, 120 breeds)
- **Transfer Learning:** MobileNetV2 from TensorFlow Hub for robust feature extraction
- **Data Augmentation:** Techniques to improve generalization and prevent overfitting
- **Evaluation Metric:** Log Loss, with probabilistic outputs for each class
- **Visualization:** Batch image grids, prediction confidence plots, and TensorBoard logs
- **Reproducibility:** Modular code for easy adaptation and deployment

---

## **Getting Started**

### **Prerequisites**
To run this notebook, you will need the following libraries installed:
* `tensorflow` (2.x)
* `tensorflow_hub`
* `tf_keras`
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
* `os`
* `IPython`

### **Installation**
You can install all necessary libraries using `pip`:
```sh
pip install tensorflow tensorflow_hub tf_keras pandas numpy matplotlib scikit-learn
```

### **Usage**
1. **Clone the repository:**  
   `git clone https://github.com/brej-29/dog-breed-classification.git`
2. **Navigate to the project directory:**  
   `cd dog-breed-classification`
3. **Open the Jupyter Notebook:**  
   `jupyter notebook DogBreedClassification.ipynb`
4. **Follow the notebook cells:**  
   - Unzip and inspect the Kaggle dataset.
   - Import core libraries (TensorFlow, TensorFlow Hub, Keras, Pandas, NumPy, Matplotlib, scikit-learn).
   - Check for GPU availability for faster training.
   - Load and explore the dataset, visualize class distributions, and inspect images.
   - Prepare labels (one-hot encoding for 120 breeds).
   - Split data into training and validation sets.
   - Preprocess images (resize, normalize) and create efficient data batches using TensorFlow's `tf.data` API.
   - Visualize batched data for sanity checks.
   - Build the model using transfer learning (MobileNetV2 from TensorFlow Hub).
   - Set up callbacks (TensorBoard, EarlyStopping) for monitoring and regularization.
   - Train the model on a subset, then on the full dataset.
   - Evaluate and visualize predictions, including confidence plots and batch visualizations.
   - Save and reload trained models for reproducibility.
   - Make predictions on test data and prepare a Kaggle-compatible CSV for submission.
   - Run inference on your own custom images and visualize results.

---

## **Data Exploration & Preprocessing**

- Load and inspect the Kaggle Dog Breed Identification dataset.
- Visualize class imbalance and breed frequencies.
- One-hot encode breed labels for multi-class classification.
- Preprocess images: resize to 224x224, normalize, and convert to tensors.
- Create efficient data pipelines with batching and shuffling for training, validation, and test sets.

---

## **Modeling & Evaluation**

- Build a transfer learning model using MobileNetV2 as a feature extractor and a dense softmax output layer.
- Compile with categorical cross-entropy loss and Adam optimizer.
- Use TensorBoard and EarlyStopping callbacks.
- Train and validate the model, visualize predictions and confidence scores.
- Prepare predictions for Kaggle submission and evaluate on custom images.

---

## **Model Deployment**

- Save trained models in HDF5 format for future inference or retraining.
- Functions provided for loading and evaluating saved models.
- Example code for making predictions on new, custom images.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contact**
If you have any questions or feedback, feel free to reach out via my [LinkedIn Profile](https://www.linkedin.com/in/brejesh-balakrishnan-7855051b9/)
