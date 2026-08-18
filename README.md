# Heart Disease Classification

## Introduction

This project is a machine learning classification project for predicting heart disease. The main goal was to train different classification models on the same dataset and compare their performance using different evaluation metrics.

A Streamlit application was also created to demonstrate the trained models through an interactive web interface. The application allows a user to upload test data, select a model, and view its performance.

## Problem Statement

Heart disease is a common health problem, and machine learning can be used to identify patterns in patient data that may help in classification.

In this project, five machine learning classification models were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

The models were evaluated using the following metrics:

* Accuracy
* AUC
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

## Dataset Description

The dataset used in this project is a cleaned and merged Heart Disease Classification dataset stored as:

`cleaned_merged_heart_dataset.csv`

The dataset contains **1,888 instances and 13 input features**, with `target` as the classification target. The dataset therefore satisfies the assignment requirement of a minimum of 12 features and 500 instances.

The input features are:

* `age` - Age of the patient
* `sex` - Sex of the patient
* `cp` - Chest pain type
* `trestbps` - Resting blood pressure
* `chol` - Serum cholesterol
* `fbs` - Fasting blood sugar
* `restecg` - Resting electrocardiographic results
* `thalachh` - Maximum heart rate achieved
* `exang` - Exercise-induced angina
* `oldpeak` - ST depression
* `slope` - Slope of the peak exercise ST segment
* `ca` - Number of major vessels
* `thal` - Thalassemia-related feature

The target column is named `target` and is used for binary classification.

The data was divided into training and testing sets using an **80:20 stratified split** with `random_state=42`. The test portion was saved separately as `test_data.csv` and is used by the Streamlit application.

## Models Used

The following classification models were trained on the dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

Logistic Regression and KNN use feature scaling through a pipeline. Random Forest is used as the ensemble model.

## Model Comparison

The following table shows the results obtained on the test dataset.

| ML Model Name       | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.7275 | 0.8324 |    0.7067 | 0.8112 |   0.7553 | 0.4566 |
| Decision Tree       |   0.9762 | 0.9763 |    0.9795 | 0.9745 |   0.9770 | 0.9523 |
| KNN                 |   0.9127 | 0.9625 |    0.9095 | 0.9235 |   0.9165 | 0.8252 |
| Naive Bayes         |   0.7090 | 0.7898 |    0.6822 | 0.8214 |   0.7454 | 0.4223 |
| Random Forest       |   0.9788 | 0.9985 |    0.9747 | 0.9847 |   0.9797 | 0.9577 |

## Observations

### Logistic Regression

Logistic Regression gave moderate results compared with the other models. It achieved an accuracy of **72.75%** and an AUC of **83.24%**. Its recall was reasonably high at **81.12%**, but its MCC score of **0.4566** was considerably lower than the tree-based models.

### Decision Tree

The Decision Tree performed very well on the test data. It achieved **97.62% accuracy**, **97.63% AUC**, and an MCC of **0.9523**. Its precision, recall, and F1 score were also very high, showing strong classification performance on the test dataset.

### KNN

KNN gave good results, achieving **91.27% accuracy** and **96.25% AUC**. Its performance was better than Logistic Regression and Naive Bayes, but it was below Decision Tree and Random Forest.

### Naive Bayes

Naive Bayes had the lowest accuracy among the five models at **70.90%**. Its recall was relatively high at **82.14%**, but its precision, F1 score, and MCC were comparatively lower.

### Random Forest

Random Forest gave the best overall performance in this experiment. It achieved **97.88% accuracy**, the highest AUC of **99.85%**, and the highest MCC of **0.9577**. It also achieved the highest recall and F1 score among the models tested.

## Overall Winner

Based on the test results, **Random Forest** was the overall best-performing model for this dataset.

It achieved the strongest overall combination of Accuracy, AUC, Recall, F1 Score, and MCC. Decision Tree was a very close second.

## Streamlit Application

The Streamlit application provides the following features:

* Upload test data in CSV format
* Select a machine learning model from a dropdown
* View Accuracy, AUC, Precision, Recall, F1 Score, and MCC
* View the confusion matrix
* View the classification report

The application is intended to demonstrate the trained models interactively using the test dataset.

## GitHub Repository

GitHub Repository:
https://github.com/cherryblossom28/ML_Assignment_2

## Project Structure

```text
ML_ASSIGNMENT_2/
│
├── app.py
├── requirements.txt
├── README.md
├── cleaned_merged_heart_dataset.csv
├── test_data.csv
├── model_results.csv
│
└── model/
    ├── train_models.py
    └── saved_models/
        ├── logistic_regression.pkl
        ├── decision_tree.pkl
        ├── knn.pkl
        ├── naive_bayes.pkl
        └── random_forest.pkl
```

## Conclusion

This project compared different classification algorithms on the same heart disease dataset and evaluated their performance using multiple classification metrics.

Among the tested models, **Random Forest performed the best overall**, while Naive Bayes and Logistic Regression produced lower scores on this test dataset.

The Streamlit application provides a simple interactive interface for uploading test data, selecting a trained model, and viewing its evaluation results.
