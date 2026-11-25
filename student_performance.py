"""
Student Performance Analysis and Score Prediction
-------------------------------------------------
A beginner-friendly Data Science project using a synthetic dataset.

Install dependencies:
    pip install -r requirements.txt

Run:
    python student_performance.py

Important:
    The bundled CSV contains synthetic demonstration data, not real student
    records. This project is for learning and portfolio demonstration only.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES = [
    "study_hours_per_day",
    "attendance_percent",
    "previous_score",
    "assignments_completed",
]
TARGET = "final_score"


def load_data(file_path: Path = DATA_FILE) -> pd.DataFrame:
    """Load the CSV file and perform basic data validation."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}. "
            "Keep student_performance.csv in the same folder as this script."
        )

    data = pd.read_csv(file_path)

    required_columns = {"student_id", *FEATURES, TARGET}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    data = data.drop_duplicates()
    data = data.dropna(subset=list(required_columns))

    if data.empty:
        raise ValueError("No usable rows remain after data validation.")

    return data


def explore_data(data: pd.DataFrame) -> None:
    """Display summaries and save exploratory data analysis charts."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n--- Dataset Preview ---")
    print(data.head())

    print("\n--- Dataset Shape ---")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    print("\n--- Descriptive Statistics ---")
    print(data.describe().round(2))

    print("\n--- Missing Values ---")
    print(data.isnull().sum())

    sns.set_theme(style="whitegrid")

    # 1. Distribution of final scores
    plt.figure(figsize=(8, 5))
    sns.histplot(data[TARGET], kde=True, color="steelblue")
    plt.title("Distribution of Final Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150)
    plt.close()

    # 2. Study hours versus final score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x="study_hours_per_day",
        y=TARGET,
    )
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours per Day")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=150)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(8, 5))
    correlation = data[FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(correlation, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nCharts saved in: {OUTPUT_DIR}")


def train_model(data: pd.DataFrame):
    """Train a Linear Regression model and evaluate its test performance."""
    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    print("\n--- Model Coefficients ---")
    for feature, coefficient in zip(FEATURES, model.coef_):
        print(f"{feature}: {coefficient:.3f}")

    return model


def predict_example(model: LinearRegression) -> None:
    """Predict a score for one hypothetical student profile."""
    example_student = pd.DataFrame(
        [
            {
                "study_hours_per_day": 4.5,
                "attendance_percent": 85.0,
                "previous_score": 72.0,
                "assignments_completed": 8,
            }
        ]
    )

    prediction = model.predict(example_student)[0]
    print("\n--- Example Prediction ---")
    print(f"Estimated final score: {prediction:.1f} / 100")
    print("This is an illustrative estimate, not a guaranteed result.")


def main() -> None:
    """Run the complete project workflow."""
    data = load_data()
    explore_data(data)
    model = train_model(data)
    predict_example(model)


if __name__ == "__main__":
    main()
