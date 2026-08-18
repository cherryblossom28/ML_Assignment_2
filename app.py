import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# Page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide"
)


# Title
st.title("❤️ Heart Disease Classification")
st.write(
    "Compare machine learning classification models "
    "for heart disease prediction."
)


# Model names and file paths
model_files = {
    "Logistic Regression": "model/saved_models/logistic_regression.pkl",
    "Decision Tree": "model/saved_models/decision_tree.pkl",
    "KNN": "model/saved_models/knn.pkl",
    "Naive Bayes": "model/saved_models/naive_bayes.pkl",
    "Random Forest": "model/saved_models/random_forest.pkl"
}


# Model selection
st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model",
    list(model_files.keys())
)


# File upload
st.header("Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(test_data)


    # Check target column
    if "target" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain a 'target' column."
        )

    else:

        # Separate features and target
        X_test = test_data.drop("target", axis=1)
        y_test = test_data["target"]


        # Load selected model
        model = joblib.load(model_files[selected_model])


        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]


        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)


        # Display metrics
        st.header("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")


        # Confusion matrix
        st.header("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        st.dataframe(cm_df)


        # Classification report
        st.header("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)


        # Selected model
        st.success(
            f"Currently selected model: {selected_model}"
        )