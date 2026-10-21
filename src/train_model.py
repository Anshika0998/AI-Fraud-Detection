from pathlib import Path
import json
import warnings

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------
# 2. CREATE MACHINE LEARNING MODELS
# --------------------------------------------------

def build_models():
    """Create the machine learning models to compare."""

    logistic_regression = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }


# --------------------------------------------------
# 3. FIND THE BEST CLASSIFICATION THRESHOLD
# --------------------------------------------------

def find_best_threshold(y_true, probabilities):
    """Choose the threshold that gives the best validation F1 score."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index])


# --------------------------------------------------
# 4. EVALUATE A MODEL
# --------------------------------------------------

def evaluate_model(model, X, y, threshold):
    """Calculate evaluation metrics for a trained model."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
        ).tolist(),
    }

    return results


# --------------------------------------------------
# 5. TRAIN AND COMPARE MODELS
# --------------------------------------------------

def main():
    """Load data, train models, evaluate them, and save the best model."""

    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 60)
    print("AI-POWERED CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    # Load the dataset.
    df = load_data()

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Separate input features and target.
    X = df.drop(columns=["Class"])
    y = df["Class"]

    feature_names = X.columns.tolist()

    # Split into development data and a final test set.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Split development data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )

    print("\nData split completed.")
    print("Training transactions:", len(X_train))
    print("Validation transactions:", len(X_val))
    print("Test transactions:", len(X_test))
    print("Fraud cases in training:", int(y_train.sum()))
    print("Fraud cases in validation:", int(y_val.sum()))
    print("Fraud cases in test:", int(y_test.sum()))

    models = build_models()

    best_model = None
    best_model_name = None
    best_threshold = 0.5
    best_average_precision = -1.0
    validation_results = {}

    # Train and compare each candidate model.
    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print("Training:", model_name)
        print("-" * 60)

        model.fit(X_train, y_train)

        validation_probabilities = model.predict_proba(X_val)[:, 1]

        threshold = find_best_threshold(
            y_val,
            validation_probabilities,
        )

        average_precision = average_precision_score(
            y_val,
            validation_probabilities,
        )

        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            threshold,
        )

        validation_results[model_name] = metrics

        print("Validation Average Precision:", round(
            average_precision, 4
        ))
        print("Validation precision:", round(
            metrics["precision"], 4
        ))
        print("Validation recall:", round(
            metrics["recall"], 4
        ))
        print("Validation F1 score:", round(
            metrics["f1_score"], 4
        ))
        print("Selected threshold:", round(threshold, 6))

        # Select the model using validation Average Precision.
        if average_precision > best_average_precision:
            best_average_precision = float(average_precision)
            best_model = model
            best_model_name = model_name
            best_threshold = threshold

    print("\n" + "=" * 60)
    print("BEST MODEL:", best_model_name)
    print("=" * 60)

    # Refit the selected model on all development data.
    # The final test set remains untouched until evaluation.
    best_model.fit(X_dev, y_dev)

    # Evaluate once on the held-out test set.
    test_results = evaluate_model(
        best_model,
        X_test,
        y_test,
        best_threshold,
    )

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\nFINAL TEST RESULTS")
    print("-" * 40)
    print("Accuracy:", round(test_results["accuracy"], 4))
    print("Precision:", round(test_results["precision"], 4))
    print("Recall:", round(test_results["recall"], 4))
    print("F1 score:", round(test_results["f1_score"], 4))
    print(
        "Average Precision:",
        round(test_results["average_precision"], 4),
    )
    print("ROC-AUC:", round(test_results["roc_auc"], 4))

    print("\nConfusion Matrix:")
    print(np.array(test_results["confusion_matrix"]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    # Save the trained model and its prediction threshold.
    model_path = MODEL_DIR / "fraud_model.joblib"

    model_bundle = {
        "model": best_model,
        "threshold": best_threshold,
        "model_name": best_model_name,
        "feature_names": feature_names,
    }

    joblib.dump(model_bundle, model_path)

    # Save the evaluation results as JSON.
    results_path = MODEL_DIR / "evaluation_results.json"

    results = {
        "selected_model": best_model_name,
        "selected_threshold": best_threshold,
        "validation_results": validation_results,
        "test_results": test_results,
    }

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Saved model to:", model_path)
    print("Saved evaluation results to:", results_path)


if __name__ == "__main__":
    main()
