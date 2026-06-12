# Handwritten Digit Recognition

This project uses the MNIST dataset to build and evaluate machine learning models for handwritten digit recognition.

The goal of this project is to practice computer vision fundamentals, neural networks, convolutional neural networks, model evaluation, and GitHub project organization.

---
## Live Demo

The Streamlit web app is available here:

[Open the Handwritten Digit Recognition App](https://handwritten-digit-recognition-9dmhuimzupedrxqm7dtzar.streamlit.app/)

## Project Overview

Handwritten digit recognition is a classic computer vision task where the goal is to classify images of handwritten digits from 0 to 9.

In this project, I used the MNIST dataset and built models step by step:

- Logistic Regression baseline model
- Dense Neural Network
- Convolutional Neural Network

The models were evaluated using accuracy, loss curves, confusion matrix, classification report, and incorrect prediction analysis.

---

## Dataset

This project uses the MNIST dataset.

The dataset contains grayscale images of handwritten digits from 0 to 9.

- Training images: 60,000
- Test images: 10,000
- Image size: 28x28 pixels
- Number of classes: 10
- Pixel values before preprocessing: 0 to 255
- Pixel values after normalization: 0 to 1

The dataset was loaded directly using TensorFlow/Keras.

---

## Project Structure

```text
handwritten-digit-recognition/
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_dense_neural_network.ipynb
│   ├── 06_cnn_model.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_model_saving_and_prediction.ipynb
│
├── reports/
│   └── figures/
│       └── .gitkeep
│
├── src/
│   └── __init__.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Models Built

### 1. Logistic Regression Baseline

The first model was a Logistic Regression classifier.

For this model, each 28x28 image was flattened into a 1D vector of 784 pixel values.

This model provides a simple baseline for comparison.

### 2. Dense Neural Network

The second model was a dense neural network built with TensorFlow/Keras.

It included:

- A `Flatten` layer
- A hidden `Dense` layer with ReLU activation
- An output `Dense` layer with softmax activation

### 3. Convolutional Neural Network

The final model was a CNN.

It included:

- `Conv2D`
- `MaxPooling2D`
- `Flatten`
- `Dense`
- Softmax output layer

The CNN is better suited for image recognition because it preserves spatial patterns in the image.

---

## Preprocessing

The MNIST images were preprocessed before model training.

The preprocessing steps included:

- Loading the MNIST dataset
- Normalizing pixel values from 0–255 to 0–1
- Flattening images for Logistic Regression
- Reshaping images to `(28, 28, 1)` for CNN input
- Keeping labels as integers from 0 to 9

For the dense neural network and CNN models, the labels were kept as integers and trained using `sparse_categorical_crossentropy`.

---

## Evaluation

The models were evaluated using:

- Test accuracy
- Test loss
- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix
- Classification report
- Incorrect prediction visualization

The confusion matrix was used to understand which digits were most often confused by the CNN.

Common digit confusions can happen between visually similar digits such as:

- 4 and 9
- 3 and 5
- 7 and 9
- 2 and 8

---

## Results

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 92.57% |
| Dense Neural Network | 97.64% |
| Convolutional Neural Network | 98.6% |

The CNN achieved the best performance because convolutional layers are designed to detect image features such as edges, curves, and shapes.

---

## Model Saving

The trained CNN model was saved using the `.keras` format.

The saved model can be loaded again using TensorFlow/Keras without retraining.

The model file is stored locally inside the `models/` folder.

Model files are ignored by Git because they can be large and should not always be uploaded to GitHub.

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/ElieAntoineGeagea/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

### 5. Open the notebooks

Start with the notebooks in this order:

```text
01_data_loading.ipynb
02_data_exploration.ipynb
03_preprocessing.ipynb
04_baseline_model.ipynb
05_dense_neural_network.ipynb
06_cnn_model.ipynb
07_model_evaluation.ipynb
08_model_saving_and_prediction.ipynb
```

---

## Key Skills Practiced

This project helped me practice:

- Python programming
- NumPy array manipulation
- Data visualization with Matplotlib
- Image preprocessing
- Logistic Regression
- Dense Neural Networks
- Convolutional Neural Networks
- TensorFlow/Keras
- Model evaluation
- Confusion matrix analysis
- Saving and loading trained models
- Git and GitHub project organization

---

## Future Improvements

Possible future improvements include:

- Add a Streamlit web application
- Allow users to upload their own handwritten digit images
- Add OpenCV preprocessing for custom images
- Improve the CNN architecture
- Add dropout to reduce overfitting
- Save evaluation figures in the `reports/figures/` folder
- Deploy the project online

---

## Project Status

Core project completed.

The project includes data loading, exploration, preprocessing, baseline modeling, dense neural network modeling, CNN modeling, evaluation, model saving, and GitHub documentation.