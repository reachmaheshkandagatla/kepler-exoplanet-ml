import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

TARGET = "koi_disposition"

st.set_page_config(
    page_title="Kepler Exoplanet Classification",
    page_icon="🪐",
    layout="wide",
)


@st.cache_resource
def load_model(model_name):
    # Load only trusted model artifacts packaged with this application.
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


@st.cache_data
def load_feature_names():
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


# Application Header
header_left, header_right = st.columns([3, 1.2])

with header_left:
    st.title("🪐 Kepler Exoplanet Classification")
    st.markdown(
        "BITS - Machine Learning Assignment-2 | "
        "Multiclass Exoplanet Classification"
    )

with header_right:
    st.markdown(
        """
        <div style="
            text-align: right;
            padding-top: 15px;
            font-size: 14px;
            line-height: 1.8;
        ">
            <b>Name:</b> Mahesh Kandagatla<br>
            <b>BITS ID:</b> 2026AJ05552<br>
            <b>Email:</b> 2026aj05552@wilp.bits-pilani.ac.in
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys()),
    index=1
)

st.sidebar.info(
    "Upload a CSV containing the 37 required features "
    "and the koi_disposition target column."
)

st.sidebar.header("Dataset Selection")

dataset_mode = st.sidebar.radio(
    "Choose Dataset",
    ["Demo Dataset", "Upload CSV"]
)

try:
    if dataset_mode == "Demo Dataset":
        df = pd.read_csv(BASE_DIR / "data" / "test_data.csv")
        st.success(f"Demo dataset loaded automatically: {len(df):,} observations")
    else:
        uploaded_file = st.file_uploader(
            "Upload Labeled Test Dataset",
            type=["csv"]
        )

        if uploaded_file is None:
            st.info("Upload a CSV to evaluate the model.")
            st.stop()

        df = pd.read_csv(uploaded_file)

    feature_names = load_feature_names()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)

    col1.metric("Observations", len(df))
    col2.metric("Columns", len(df.columns))

    required_columns = feature_names + [TARGET]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    X = df[feature_names].copy()
    y_true = df[TARGET]

    if y_true.isna().any():
        st.error("Target column contains missing labels.")
        st.stop()

    for column in feature_names:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        )

    model = load_model(selected_model)

    unknown_labels = set(y_true.unique()) - set(model.classes_)

    if unknown_labels:
        st.error(
            f"Unknown target labels: {sorted(unknown_labels)}"
        )
        st.stop()

    if len(set(y_true)) < len(model.classes_):
        st.error(
            "Evaluation requires all three target classes "
            "to be present for the configured multiclass AUC."
        )
        st.stop()

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true,
            y_proba,
            labels=model.classes_,
            multi_class="ovr",
            average="macro",
        ),
        "Precision": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "F1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    st.success(
        f"Evaluation completed using {selected_model}! "
        "To evaluate other models, use the 'Select Classification Model' "
        "dropdown in the left sidebar."
    )

    st.subheader("Model Performance")

    cols = st.columns(3)

    for index, (name, value) in enumerate(metrics.items()):
        cols[index % 3].metric(name, f"{value:.4f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=model.classes_,
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax,
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        plt.xticks(rotation=30)
        plt.yticks(rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            labels=model.classes_,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True,
        )
        # Dynamic Model Performance Analysis
    st.subheader("Model Performance Analysis")

    # Calculate recall for the CANDIDATE class
    candidate_mask = y_true == "CANDIDATE"
    candidate_total = candidate_mask.sum()

    candidate_correct = (
        (y_pred == "CANDIDATE") & candidate_mask.to_numpy()
    ).sum()

    candidate_recall = (
        candidate_correct / candidate_total
        if candidate_total > 0 else 0
    )

    # Identify class imbalance
    class_distribution = y_true.value_counts(normalize=True)
    majority_class = class_distribution.idxmax()
    majority_percentage = class_distribution.max() * 100

    # Generate model-specific analysis
    analysis = (
        f"{selected_model} achieves {metrics['Accuracy']:.2%} accuracy "
        f"and {metrics['F1']:.4f} Macro F1. "
        f"CANDIDATE recall is {candidate_recall:.2%}. "
        f"The {majority_class} class represents "
        f"{majority_percentage:.1f}% of observations. "
    )

    # Interpret candidate recall
    if candidate_total == 0:
        analysis += (
            "No CANDIDATE observations are available "
            "for class-specific evaluation."
        )
    elif candidate_recall < 0.40:
        analysis += (
            "Low candidate recall indicates difficulty "
            "identifying potential exoplanets."
        )
    elif candidate_recall < 0.70:
        analysis += (
            "Moderate candidate recall suggests "
            "room for classification improvement."
        )
    else:
        analysis += (
            "Strong candidate recall indicates "
            "effective identification of potential exoplanets."
        )

    st.info(analysis)
    

    st.divider()

    st.subheader("Prediction Results")

    results_df = df.copy()
    results_df["Predicted_Class"] = y_pred
    results_df["Correct_Prediction"] = (
        y_true.to_numpy() == y_pred
    )

    st.dataframe(
        results_df[
            [TARGET, "Predicted_Class", "Correct_Prediction"]
        ].head(50),
        use_container_width=True,
    )

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        data=csv,
        file_name="kepler_predictions.csv",
        mime="text/csv",
    )

except Exception as error:
    st.error(f"Evaluation failed: {error}")
