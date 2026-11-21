import pandas as pd

# 1. Load CSV file
df = pd.read_csv("StudentPerformanceFactors.csv")

print("Original Dataset:")
print(df.head())

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Check duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 4. Remove duplicate rows
df = df.drop_duplicates()

# 5. Clean column names
df.columns = df.columns.str.strip()

# 6. Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# 7. Fill missing values
for col in df.columns:

    # Numeric columns -> fill with median
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

    # Text columns -> fill with mode
    else:
        if df[col].isnull().any():
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])

# 8. Save cleaned CSV
df.to_csv("StudentPerformanceFactors_cleaned.csv", index=False)

# 9. Final output
print("\nCleaning Completed Successfully!")
print("Final Dataset Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nFile saved as StudentPerformanceFactors_cleaned.csv")