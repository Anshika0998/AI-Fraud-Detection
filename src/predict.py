from pathlib import Path
import argparse

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"


def load_model():
    """Load the saved fraud detection model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run 'python src/train_model.py' first."
        )

    bundle = joblib.load(MODEL_PATH)

    return bundle


def predict_transactions(input_path, output_path=None):
    """Predict fraud probabilities for transactions in a CSV file."""

    bundle = load_model()

    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_names = bundle["feature_names"]

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Load transactions
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The input CSV file is empty.")

    # The target column is optional in prediction files.
    # If present, do not use it as a model input.
    X = df.drop(columns=["Class"], errors="ignore")

    # Check for missing required columns.
    missing_columns = [
        column
        for column in feature_names
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Ensure the feature order matches training.
    X = X[feature_names]

    # Predict fraud probabilities.
    probabilities = model.predict_proba(X)[:, 1]

    # Apply the threshold selected during validation.
    predictions = (probabilities >= threshold).astype(int)

    # Create output
    results = df.copy()

    results["Fraud_Probability"] = probabilities
    results["Prediction"] = predictions

    results["Prediction_Label"] = results["Prediction"].map({
        0: "Legitimate",
        1: "Potential Fraud",
    })

    # Save results
    if output_path is None:
        output_path = (
            PROJECT_ROOT / "predictions.csv"
        )
    else:
        output_path = Path(output_path)

    results.to_csv(output_path, index=False)

    print("\n========== FRAUD PREDICTION RESULTS ==========")
    print("Total transactions:", len(results))
    print(
        "Potentially fraudulent transactions:",
        int(predictions.sum()),
    )
    print(
        "Legitimate transactions:",
        int((predictions == 0).sum()),
    )
    print("Results saved to:", output_path)

    print("\nFirst five predictions:")
    print(
        results[
            [
                "Fraud_Probability",
                "Prediction",
                "Prediction_Label",
            ]
        ].head()
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict potentially fraudulent transactions."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input transaction CSV file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the prediction results CSV.",
    )

    args = parser.parse_args()

    predict_transactions(
        input_path=args.input,
        output_path=args.output,
    )