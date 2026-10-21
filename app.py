from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"


st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# LOAD MODEL AND RESULTS
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained model once and cache it."""

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def load_evaluation_results():
    """Load model evaluation metrics."""

    if not RESULTS_PATH.exists():
        return None

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


bundle = load_model()
evaluation_results = load_evaluation_results()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ Fraud Detection")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Predict Transactions",
        "Model Performance",
        "About the Project",
    ],
)


# --------------------------------------------------
# CHECK WHETHER MODEL EXISTS
# --------------------------------------------------

if bundle is None:

    st.error(
        "The trained model was not found."
    )

    st.info(
        "Open your VS Code terminal and run:\n\n"
        "`python src/train_model.py`\n\n"
        "After training finishes, restart or refresh this application."
    )

    st.stop()


model = bundle["model"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]
feature_names = bundle["feature_names"]


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.title("🛡️ AI-Powered Financial Fraud Detection")

    st.write(
        "An interactive machine learning application "
        "for identifying potentially fraudulent credit card transactions."
    )

    st.divider()

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Selected Model",
            model_name,
        )

    with col2:
        st.metric(
            "Model Features",
            len(feature_names),
        )

    with col3:
        st.metric(
            "Decision Threshold",
            f"{threshold:.4f}",
        )

    st.divider()

    st.subheader("Model Evaluation")

    if evaluation_results:

        test_metrics = evaluation_results.get(
            "test_results",
            {},
        )

        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Precision",
            f"{test_metrics.get('precision', 0):.2%}",
        )

        metric_columns[1].metric(
            "Recall",
            f"{test_metrics.get('recall', 0):.2%}",
        )

        metric_columns[2].metric(
            "F1 Score",
            f"{test_metrics.get('f1_score', 0):.2%}",
        )

        metric_columns[3].metric(
            "ROC-AUC",
            f"{test_metrics.get('roc_auc', 0):.2%}",
        )

        st.caption(
            "Metrics above are calculated on the held-out test set. "
            "Precision, recall, and F1 use the selected decision threshold."
        )

    else:

        st.warning(
            "Evaluation results are unavailable. "
            "Run the training script to generate them."
        )

    st.divider()

    st.subheader("How This Application Works")

    st.markdown(
        """
        1. **Data input:** Upload a CSV file containing transaction features.
        2. **Preprocessing:** The trained pipeline handles missing values
           and any model-specific transformations.
        3. **Prediction:** The model estimates the probability of fraud.
        4. **Classification:** The probability is compared with the
           selected decision threshold.
        5. **Results:** View and download the predictions.
        """
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predict Transactions":

    st.title("🔍 Transaction Fraud Prediction")

    st.write(
        "Upload a CSV file containing transaction data "
        "to generate fraud predictions."
    )

    st.info(
        "Your CSV must contain the same input features used during training: "
        "Time, V1–V28, and Amount. The Class column is optional and "
        "will not be used as an input feature."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV file is empty.")
                st.stop()

            # Avoid treating the actual answer as an input.
            X = df.drop(
                columns=["Class"],
                errors="ignore",
            )

            missing_columns = [
                column
                for column in feature_names
                if column not in X.columns
            ]

            if missing_columns:

                st.error(
                    "Your CSV is missing these required columns: "
                    + ", ".join(missing_columns)
                )

                st.stop()

            # Keep features in the correct order.
            X = X[feature_names]

            if len(X) > 100000:

                st.error(
                    "This demo accepts at most 100,000 transactions "
                    "per upload. Split larger files into smaller CSV files."
                )

                st.stop()

            # Predict button
            if st.button(
                "Analyze Transactions",
                type="primary",
            ):

                with st.spinner("Analyzing transactions..."):

                    probabilities = model.predict_proba(X)[:, 1]

                    predictions = (
                        probabilities >= threshold
                    ).astype(int)

                    results = df.copy()

                    results["Fraud_Probability"] = probabilities

                    results["Prediction"] = predictions

                    results["Prediction_Label"] = (
                        results["Prediction"].map({
                            0: "Legitimate",
                            1: "Potential Fraud",
                        })
                    )

                st.success("Analysis completed!")

                total = len(results)
                fraud_count = int(predictions.sum())
                legitimate_count = total - fraud_count

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Transactions Analyzed",
                    f"{total:,}",
                )

                col2.metric(
                    "Potential Fraud",
                    f"{fraud_count:,}",
                )

                col3.metric(
                    "Legitimate Predictions",
                    f"{legitimate_count:,}",
                )

                st.divider()

                st.subheader("Prediction Distribution")

                chart_data = pd.DataFrame({
                    "Category": [
                        "Legitimate",
                        "Potential Fraud",
                    ],
                    "Count": [
                        legitimate_count,
                        fraud_count,
                    ],
                })

                fig = px.bar(
                    chart_data,
                    x="Category",
                    y="Count",
                    title="Predicted Transaction Categories",
                    text="Count",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.subheader("Transaction Results")

                st.dataframe(
                    results.head(1000),
                    use_container_width=True,
                )

                if len(results) > 1000:
                    st.caption(
                        "Showing the first 1,000 rows in the preview. "
                        "The download contains all analyzed transactions."
                    )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

                st.warning(
                    "These are model predictions, not confirmed fraud cases. "
                    "A real financial institution would need additional "
                    "validation and a human review process."
                )

        except (
            pd.errors.ParserError,
            UnicodeDecodeError,
        ):

            st.error(
                "The uploaded file could not be read as a CSV. "
                "Please upload a valid CSV file."
            )

        except Exception as error:

            st.error(
                f"An error occurred while processing the file: {error}"
            )


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    if not evaluation_results:

        st.warning(
            "No evaluation results were found. "
            "Run the training script first."
        )

    else:

        st.subheader("Selected Model")

        st.write(
            f"**Model:** {evaluation_results['selected_model']}"
        )

        st.write(
            f"**Threshold:** "
            f"{evaluation_results['selected_threshold']:.6f}"
        )

        st.divider()

        st.subheader("Held-Out Test Metrics")

        test_metrics = evaluation_results["test_results"]

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "average_precision",
            "roc_auc",
        ]

        metrics_df = pd.DataFrame({
            "Metric": metric_names,
            "Score": [
                test_metrics[name]
                for name in metric_names
            ],
        })

        fig = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            title="Test Set Evaluation Metrics",
            text_auto=".3f",
        )

        fig.update_yaxes(range=[0, 1])

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader("Confusion Matrix")

        matrix = test_metrics["confusion_matrix"]

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        )

        st.dataframe(
            matrix_df,
            use_container_width=True,
        )

        st.caption(
            "The confusion matrix shows how many legitimate and fraudulent "
            "transactions were correctly or incorrectly classified."
        )

        st.divider()

        st.subheader("Model Comparison")

        validation_results = evaluation_results.get(
            "validation_results",
            {},
        )

        if validation_results:

            comparison_rows = []

            for name, metrics in validation_results.items():

                comparison_rows.append({
                    "Model": name,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"],
                    "Average Precision": metrics["average_precision"],
                    "ROC-AUC": metrics["roc_auc"],
                })

            comparison_df = pd.DataFrame(
                comparison_rows
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
            )

            st.caption(
                "The model comparison uses validation results. "
                "The final test set is reserved for evaluating the "
                "selected model."
            )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About the Project":

    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ## AI-Powered Financial Fraud Detection System

        This project explores how machine learning can help identify
        potentially fraudulent credit card transactions.

        ### Technologies Used

        - Python
        - Pandas and NumPy
        - Scikit-learn
        - Logistic Regression
        - Random Forest
        - Streamlit
        - Plotly
        - Joblib

        ### Dataset

        European credit card transactions from the Kaggle
        Credit Card Fraud Detection dataset.

        The dataset contains anonymized transaction features, including
        Time, Amount, and V1–V28.

        ### Evaluation

        Because fraudulent transactions are rare, accuracy alone is
        not sufficient to evaluate this project.

        Precision, recall, F1 score, Average Precision, ROC-AUC,
        and the confusion matrix provide a more useful picture
        of model performance.

        ### Important Limitation

        This is an educational machine learning project. Its predictions
        must not be treated as definitive fraud determinations or used
        as the sole basis for real financial decisions.
        """
    )

    st.link_button(
        "View the Kaggle Dataset",
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    )

    st.info(
        "Future improvements could include explainable AI, "
        "drift monitoring, improved threshold selection, "
        "and an API for real-time predictions."
    )