 # Heart Disease Classification

## Introduction

This project is a machine learning classification project for predicting heart disease. The main goal was to train different classification models on the same dataset and compare their performance using different evaluation metrics.

I also created a Streamlit application so that the trained models can be tested through a simple web interface. The application allows a user to upload test data, select a model, and view its performance.

## Problem Statement

Heart disease is a common health problem, and machine learning can be used to identify patterns in patient data that may help in classification.

In this project, five machine learning classification models were implemented and compared:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Random Forest

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## Dataset

The dataset used in this project is a cleaned and merged heart disease dataset stored as:

`cleaned_merged_heart_dataset.csv`

The target column is named `target` and is used for the classification task.

The data was divided into training and testing sets using an 80:20 split. Stratification and `random_state=42` were used while splitting the data. The test portion was saved separately as `test_data.csv` for use in the Streamlit application.

## Models Used

The following models were trained on the dataset:

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

Logistic Regression gave moderate results compared with the other models. Its accuracy was about 72.75%, while the AUC was 83.24%. The recall was reasonably high, but the overall MCC score was much lower than the tree-based models.

### Decision Tree

The Decision Tree performed very well on the test data. It achieved 97.62% accuracy and an MCC of 0.9523. Its precision, recall, and F1 score were also very high, showing that the model was able to classify the test samples effectively.

### KNN

KNN gave good results, with an accuracy of 91.27% and an AUC of 96.25%. Its performance was better than Logistic Regression and Naive Bayes, but it was still below Decision Tree and Random Forest.

### Naive Bayes

Naive Bayes had the lowest accuracy among the five models at about 70.90%. Its recall was relatively high at 82.14%, but its precision, F1 score, and MCC were comparatively lower.

### Random Forest

Random Forest gave the best overall performance in this experiment. It achieved 97.88% accuracy, the highest AUC of 99.85%, and the highest MCC of 0.9577. It also had the highest recall and F1 score among the models tested.

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

This project helped compare different classification algorithms on the same heart disease dataset and understand how their performance changes across different evaluation metrics.

Among the tested models, Random Forest performed the best overall, while Naive Bayes and Logistic Regression produced lower scores on this test dataset. The Streamlit application provides a simple way to test the models and view their results interactively.
