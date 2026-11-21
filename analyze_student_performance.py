"""
Student Performance Data Analysis
Dataset: StudentPerformanceFactors_cleaned.csv

Run:
    python analyze_student_performance.py

Requirements:
    pip install pandas matplotlib
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).with_name("StudentPerformanceFactors_cleaned.csv")
OUTPUT_DIR = Path(__file__).with_name("analysis_plots")


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        print("Place StudentPerformanceFactors_cleaned.csv beside this script.")
        return

    df = pd.read_csv(DATA_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("STUDENT PERFORMANCE DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset overview
    print("\\n[Dataset Overview]")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\\nColumn names:")
    print(list(df.columns))
    print("\\nData types:")
    print(df.dtypes)
    print("\\nFirst 5 rows:")
    print(df.head())

    # 2. Missing values
    print("\\n[Missing Values]")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found.")

    # 3. Summary statistics
    print("\\n[Numeric Summary Statistics]")
    print(df.describe())

    # 4. Categorical frequency counts
    categorical_cols = df.select_dtypes(include="object").columns
    print("\\n[Categorical Value Counts]")
    for column in categorical_cols:
        print(f"\\n{column}:")
        print(df[column].value_counts(dropna=False))

    # 5. Exam score analysis
    if "Exam_Score" in df.columns:
        scores = df["Exam_Score"]
        print("\\n[Exam Score Insights]")
        print(f"Average exam score: {scores.mean():.2f}")
        print(f"Median exam score: {scores.median():.2f}")
        print(f"Minimum exam score: {scores.min()}")
        print(f"Maximum exam score: {scores.max()}")
        print(f"Standard deviation: {scores.std():.2f}")

        # Histogram of exam scores
        plt.figure(figsize=(8, 5))
        plt.hist(scores.dropna(), bins=15, edgecolor="black")
        plt.title("Distribution of Exam Scores")
        plt.xlabel("Exam Score")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "exam_score_distribution.png", dpi=300)
        plt.close()

    # 6. Relationship between study hours and exam scores
    if {"Hours_Studied", "Exam_Score"}.issubset(df.columns):
        correlation = df["Hours_Studied"].corr(df["Exam_Score"])
        print(f"\\nCorrelation (Hours Studied vs Exam Score): {correlation:.3f}")

        plt.figure(figsize=(8, 5))
        plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.5)
        plt.title("Hours Studied vs Exam Score")
        plt.xlabel("Hours Studied")
        plt.ylabel("Exam Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "study_hours_vs_exam_score.png", dpi=300)
        plt.close()

    # 7. Attendance vs exam scores
    if {"Attendance", "Exam_Score"}.issubset(df.columns):
        correlation = df["Attendance"].corr(df["Exam_Score"])
        print(f"Correlation (Attendance vs Exam Score): {correlation:.3f}")

        plt.figure(figsize=(8, 5))
        plt.scatter(df["Attendance"], df["Exam_Score"], alpha=0.5)
        plt.title("Attendance vs Exam Score")
        plt.xlabel("Attendance")
        plt.ylabel("Exam Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "attendance_vs_exam_score.png", dpi=300)
        plt.close()

    # 8. Average exam score by selected categorical factors
    if "Exam_Score" in df.columns:
        selected_factors = [
            "Motivation_Level",
            "Parental_Involvement",
            "Access_to_Resources",
            "Internet_Access",
            "School_Type",
            "Peer_Influence",
        ]

        for factor in selected_factors:
            if factor in df.columns:
                averages = df.groupby(factor, dropna=False)["Exam_Score"].mean().sort_values()
                plt.figure(figsize=(8, 5))
                averages.plot(kind="bar", edgecolor="black")
                plt.title(f"Average Exam Score by {factor.replace('_', ' ')}")
                plt.xlabel(factor.replace("_", " "))
                plt.ylabel("Average Exam Score")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                safe_name = factor.lower()
                plt.savefig(OUTPUT_DIR / f"average_score_by_{safe_name}.png", dpi=300)
                plt.close()
                print(f"\\nAverage Exam Score by {factor}:")
                print(averages.round(2))

    print(f"\\nAnalysis complete. Charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
