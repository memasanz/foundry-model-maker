"""Score new rows with a model.pkl produced by the ML-trainer agent.

IMPORTANT: the Code Interpreter sandbox trains with scikit-learn 1.1.3, and
scikit-learn pickles are version-sensitive. Load the model in an environment
built from requirements-loader.txt (scikit-learn==1.1.3, Python 3.8-3.11),
otherwise unpickling may fail with an "incompatible dtype" error.

    python predict.py data/model.pkl data/housing.csv
"""

import sys

import joblib
import pandas as pd


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python predict.py <model.pkl> <rows.csv>")
        sys.exit(1)

    model = joblib.load(sys.argv[1])
    df = pd.read_csv(sys.argv[2])

    # Drop the target column if the scoring file still contains it.
    for candidate in ("price", "target", "label", "y"):
        if candidate in df.columns:
            df = df.drop(columns=[candidate])

    predictions = model.predict(df)
    for i, value in enumerate(predictions):
        print(f"row {i}: {value:.2f}")


if __name__ == "__main__":
    main()
