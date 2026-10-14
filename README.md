# Traffic Sign Classification using CNN

## Project Overview

This project implements an automatic traffic sign classification system using a Convolutional Neural Network (CNN) developed with TensorFlow and Keras.

The system is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and can recognize **43 different traffic sign classes**.

A Streamlit web application is also provided to allow users to upload a traffic sign image and obtain a prediction with the corresponding confidence score.

---

## Features

* Traffic sign classification using CNN
* 43 traffic sign classes
* Image preprocessing and normalization
* Training and validation
* Model evaluation
* Classification report
* Confusion matrix
* Wrong prediction analysis
* Top predictions with confidence scores
* Streamlit web application
* Prediction from uploaded images
* Inference time measurement

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* OpenCV
* Matplotlib
* Streamlit

---

## Project Structure

```text
traffic-sign-classification/
│
├── app/
│   └── app.py
│
├── data/
│   └── dataset/
│       ├── Train.csv
│       ├── Test.csv
│       ├── Meta.csv
│       └── Test/
│
├── models/
│   └── traffic_sign_model.keras
│
├── src/
│   ├── __init__.py
│   ├── classification_report.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── visualize.py
│
├── visualizations/
│   ├── classification_report.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── wrong_predictions.png
│
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The dataset contains images belonging to 43 different traffic sign classes.

The data is divided into training and testing data. A validation subset is created from the training data during the training process.
 
 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
---

## Model Architecture

The classification model is a Convolutional Neural Network composed of:

* Input layer: `32 × 32 × 3`
* Convolutional layer: 32 filters
* Max Pooling
* Convolutional layer: 64 filters
* Max Pooling
* Convolutional layer: 128 filters
* Max Pooling
* Flatten layer
* Dense layer: 128 neurons
* Dropout: 0.5
* Output layer: 43 classes with Softmax activation

The model uses:

* Adam optimizer
* Sparse Categorical Crossentropy loss
* Accuracy as the main evaluation metric

---

## Training

The model was trained using:

* Image size: `32 × 32`
* Batch size: `32`
* Maximum epochs: `20`
* Learning rate: `0.001`
* Validation split: `20%`

Early stopping and model checkpointing were used to preserve the best model.

The best validation accuracy achieved during training was approximately:

**99.34%**

---

## Evaluation Results

The trained model was evaluated on **12,630 test images**.

### Test Performance

| Metric            |     Result |
| ----------------- | ---------: |
| Test Accuracy     | **95.59%** |
| Macro F1-score    |   **0.93** |
| Weighted F1-score |   **0.96** |
| Test Samples      | **12,630** |
| Number of Classes |     **43** |

The results show that the CNN can successfully classify most traffic sign classes.

Some classes have lower F1-scores than others, which indicates that certain traffic signs are more difficult to distinguish.

---

## Error Analysis

The project includes several tools for analyzing model errors:

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

for each of the 43 classes.

### Confusion Matrix

The confusion matrix helps identify classes that are frequently confused with each other.

### Wrong Predictions

The project also generates examples of incorrectly classified images to visually analyze model errors.

The error analysis indicates that some classes are more difficult to classify, particularly classes with lower F1-scores or fewer test samples.

---

## Streamlit Application

The project includes a web application developed with Streamlit.

The application allows the user to:

1. Upload a traffic sign image.
2. Display the input image.
3. Predict the traffic sign class.
4. Display the confidence score.
5. Display the traffic sign category.
6. Display the top predictions.
7. Display the inference time.

---

## Installation

### 1. Clone the GitHub repository

First, clone the project from GitHub:

```bash
git clone https://github.com/YOUR-USERNAME/traffic-sign-classification.git
```

Then enter the project directory:

```bash
cd traffic-sign-classification
```

> Replace `YOUR-USERNAME` with the GitHub username that owns the repository.

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

From the project root directory:

```bash
python -m streamlit run app/app.py
```

The application will open in the browser.

---

## Training the Model

If you want to train the model again:

```bash
python -m src.train
```

The trained model will be saved in:

```text
models/traffic_sign_model.keras
```

> The repository already contains the trained model if it has been uploaded to GitHub. Retraining is therefore not necessary just to run the application.

---

## Evaluating the Model

To evaluate the model on the test dataset:

```bash
python -m src.evaluate
```

---

## Generating the Classification Report

Run:

```bash
python -m src.classification_report
```

The generated files are saved in:

```text
visualizations/
```

including:

```text
classification_report.txt
classification_report.csv
```

---

## Generating Visualizations

To generate the confusion matrix and wrong prediction visualizations:

```bash
python -m src.visualize
```

The results are saved in:

```text
visualizations/
```

---

## Making a Prediction

A prediction can also be performed using the prediction script:

```bash
python -m src.predict
```

The Streamlit application provides a more interactive way to perform image predictions.

---

## Limitations

Although the model achieves high accuracy on the GTSRB test dataset, some limitations remain:

* Some traffic signs are visually similar.
* Some classes contain fewer samples than others.
* Images uploaded by users may differ from the training dataset.
* Changes in lighting, angle, resolution, background, or image quality can affect predictions.
* The model is designed specifically for the traffic sign classes represented in the GTSRB dataset.

---

## Future Improvements

Possible improvements include:

* Data augmentation
* Increasing the number of training images
* Improving the CNN architecture
* Using transfer learning
* Testing additional real-world images
* Improving robustness to different lighting and viewing angles
* Adding more traffic sign datasets
* Deploying the application online

---

## Conclusion

This project demonstrates how deep learning can be used to automatically recognize traffic signs.

A CNN was trained using the GTSRB dataset and achieved a test accuracy of **95.59%**.

The project also includes an interactive Streamlit application that allows users to upload traffic sign images and receive predictions with confidence scores.

The combination of model training, evaluation, error analysis, and an interactive web application provides a complete end-to-end traffic sign classification system.
