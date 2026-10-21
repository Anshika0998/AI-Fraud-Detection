from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def load_data():
    """Load the credit card transaction dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place creditcard.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a 'Class' column.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["Class"].isna().any():
        raise ValueError("The Class column contains missing values.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("The Class column must contain only 0 and 1.")

    return df


def inspect_data(df):
    """Display basic information about the dataset."""

    print("\n========== DATASET OVERVIEW ==========")
    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTransaction class distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentages:")
    print(df["Class"].value_counts(normalize=True) * 100)


def prepare_data(df, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n========== TRAIN-TEST SPLIT ==========")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Fraud cases in training:", int(y_train.sum()))
    print("Fraud cases in testing:", int(y_test.sum()))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataframe = load_data()
    inspect_data(dataframe)
    prepare_data(dataframe)