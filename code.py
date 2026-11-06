# ============================================================
# 281K PATIENT DATASET FOR MULTI-DISEASE PREDICTION
# COMPLETE DATA ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\dilip\OneDrive\文件\ML PROJECT DATASETS\281K Patient Dataset for Multi-Disease Prediction\Data Warehouse Multiclass.csv"
)

print("==================================================")
print("     281K PATIENT DATASET FOR MULTI-DISEASE")
print("==================================================")


# ============================================================
# 2. DISPLAY COMPLETE DATASET
# ============================================================

print("\n1. COMPLETE DATASET:")

print(df.to_string())


# ============================================================
# 3. FIRST 10 RECORDS
# ============================================================

print("\n\n2. FIRST 10 PATIENT RECORDS:")

print(df.head(10))


# ============================================================
# 4. LAST 10 RECORDS
# ============================================================

print("\n\n3. LAST 10 PATIENT RECORDS:")

print(df.tail(10))


# ============================================================
# 5. DATASET SHAPE
# ============================================================

print("\n\n4. DATASET SHAPE:")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ============================================================
# 6. COLUMN NAMES
# ============================================================

print("\n\n5. COLUMN NAMES:")

for i, column in enumerate(df.columns, start=1):
    print(i, ".", column)


# ============================================================
# 7. DATA TYPES
# ============================================================

print("\n\n6. DATA TYPES:")

print(df.dtypes)


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

print("\n\n7. DATASET INFORMATION:")

df.info()


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n\n8. MISSING VALUES:")

print(df.isnull().sum())


# ============================================================
# 10. TOTAL MISSING VALUES
# ============================================================

print("\n\n9. TOTAL MISSING VALUES:")

print(df.isnull().sum().sum())


# ============================================================
# 11. DUPLICATE RECORDS
# ============================================================

print("\n\n10. DUPLICATE RECORDS:")

print(df.duplicated().sum())


# ============================================================
# 12. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nDataset after removing duplicates:")

print(df.shape)


# ============================================================
# 13. STATISTICAL INFORMATION
# ============================================================

print("\n\n11. STATISTICAL INFORMATION:")

print(df.describe())


# ============================================================
# 14. UNIQUE VALUES
# ============================================================

print("\n\n12. UNIQUE VALUES IN EACH COLUMN:")

for column in df.columns:

    print(
        column,
        "=",
        df[column].nunique()
    )


# ============================================================
# 15. GENDER DISTRIBUTION
# ============================================================

print("\n\n13. GENDER DISTRIBUTION:")

print(
    df["gender"].value_counts()
)


# ============================================================
# 16. AGE LEVEL DISTRIBUTION
# ============================================================

print("\n\n14. AGE LEVEL DISTRIBUTION:")

print(
    df["age_level"].value_counts()
)


# ============================================================
# 17. BMI LEVEL DISTRIBUTION
# ============================================================

print("\n\n15. BMI LEVEL DISTRIBUTION:")

print(
    df["bmi_level"].value_counts()
)


# ============================================================
# 18. SMOKING DISTRIBUTION
# ============================================================

print("\n\n16. SMOKING DISTRIBUTION:")

print(
    df["smoking"].value_counts()
)


# ============================================================
# 19. DIABETES DISTRIBUTION
# ============================================================

print("\n\n17. DIABETES DISTRIBUTION:")

print(
    df["diabetes"].value_counts()
)


# ============================================================
# 20. HYPERTENSION DISTRIBUTION
# ============================================================

print("\n\n18. HYPERTENSION DISTRIBUTION:")

print(
    df["hypertension"].value_counts()
)


# ============================================================
# 21. HEART DISEASE DISTRIBUTION
# ============================================================

print("\n\n19. HEART DISEASE DISTRIBUTION:")

print(
    df["heart_disease"].value_counts()
)


# ============================================================
# 22. MULTI-DISEASE LABEL DISTRIBUTION
# ============================================================

print("\n\n20. MULTI-DISEASE LABEL DISTRIBUTION:")

print(
    df["label"].value_counts()
)


# ============================================================
# 23. SUBLABEL DISTRIBUTION
# ============================================================

print("\n\n21. SUBLABEL DISTRIBUTION:")

print(
    df["sublabel"].value_counts()
)


# ============================================================
# 24. SOURCE DATASET DISTRIBUTION
# ============================================================

print("\n\n22. SOURCE DATASET DISTRIBUTION:")

print(
    df["source_dataset"].value_counts()
)


# ============================================================
# 25. DISEASE FLAGS
# ============================================================

print("\n\n23. DISEASE FLAGS:")

print(
    df["disease_flags"].value_counts()
)


# ============================================================
# 26. AVERAGE PATIENT AGE
# ============================================================

print("\n\n24. AVERAGE PATIENT AGE:")

print(
    df["age"].mean()
)


# ============================================================
# 27. AVERAGE BMI
# ============================================================

print("\n\n25. AVERAGE BMI:")

print(
    df["bmi"].mean()
)


# ============================================================
# 28. AVERAGE GLUCOSE
# ============================================================

print("\n\n26. AVERAGE GLUCOSE:")

print(
    df["glucose"].mean()
)


# ============================================================
# 29. AVERAGE CHOLESTEROL
# ============================================================

print("\n\n27. AVERAGE CHOLESTEROL:")

print(
    df["cholesterol"].mean()
)


# ============================================================
# 30. AVERAGE BLOOD PRESSURE
# ============================================================

print("\n\n28. AVERAGE SYSTOLIC BP:")

print(
    df["systolic_bp"].mean()
)

print("\nAverage Diastolic BP:")

print(
    df["diastolic_bp"].mean()
)


# ============================================================
# 31. AVERAGE HEART RATE
# ============================================================

print("\n\n29. AVERAGE HEART RATE:")

print(
    df["heart_rate"].mean()
)


# ============================================================
# 32. AVERAGE HbA1c
# ============================================================

print("\n\n30. AVERAGE HbA1c:")

print(
    df["HbA1c_level"].mean()
)


# ============================================================
# 33. HIGHEST BMI
# ============================================================

print("\n\n31. HIGHEST BMI:")

print(
    df["bmi"].max()
)


# ============================================================
# 34. LOWEST BMI
# ============================================================

print("\n\n32. LOWEST BMI:")

print(
    df["bmi"].min()
)


# ============================================================
# 35. OLDEST PATIENT
# ============================================================

print("\n\n33. OLDEST PATIENT AGE:")

print(
    df["age"].max()
)


# ============================================================
# 36. YOUNGEST PATIENT
# ============================================================

print("\n\n34. YOUNGEST PATIENT AGE:")

print(
    df["age"].min()
)


# ============================================================
# 37. NUMERICAL COLUMN STATISTICS
# ============================================================

print("\n\n35. NUMERICAL COLUMN STATISTICS:")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(
    df[numeric_columns].describe()
)


# ============================================================
# 38. CATEGORICAL COLUMN INFORMATION
# ============================================================

print("\n\n36. CATEGORICAL COLUMN INFORMATION:")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print("\n--------------------------------------")
    print(column)
    print("--------------------------------------")

    print(
        df[column].value_counts().head(10)
    )


# ============================================================
# GRAPHS
# ============================================================


# ============================================================
# 39. GENDER GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

df["gender"].value_counts().plot(
    kind="bar"
)

plt.title("Patient Gender Distribution")

plt.xlabel("Gender")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 40. AGE LEVEL GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

df["age_level"].value_counts().plot(
    kind="bar"
)

plt.title("Patient Age Level Distribution")

plt.xlabel("Age Level")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 41. BMI DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["bmi"],
    bins=30
)

plt.title("BMI Distribution")

plt.xlabel("BMI")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.show()


# ============================================================
# 42. AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=30
)

plt.title("Patient Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.show()


# ============================================================
# 43. GLUCOSE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["glucose"],
    bins=30
)

plt.title("Glucose Distribution")

plt.xlabel("Glucose")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.show()


# ============================================================
# 44. CHOLESTEROL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["cholesterol"],
    bins=30
)

plt.title("Cholesterol Distribution")

plt.xlabel("Cholesterol")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.show()


# ============================================================
# 45. DIABETES GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

df["diabetes"].value_counts().plot(
    kind="bar"
)

plt.title("Diabetes Distribution")

plt.xlabel("Diabetes")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 46. HYPERTENSION GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

df["hypertension"].value_counts().plot(
    kind="bar"
)

plt.title("Hypertension Distribution")

plt.xlabel("Hypertension")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 47. HEART DISEASE GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

df["heart_disease"].value_counts().plot(
    kind="bar"
)

plt.title("Heart Disease Distribution")

plt.xlabel("Heart Disease")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 48. MULTI-DISEASE LABEL GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

df["label"].value_counts().head(15).plot(
    kind="bar"
)

plt.title("Top Multi-Disease Labels")

plt.xlabel("Disease Label")

plt.ylabel("Number of Patients")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 49. SMOKING GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

df["smoking"].value_counts().plot(
    kind="bar"
)

plt.title("Smoking Status Distribution")

plt.xlabel("Smoking Status")

plt.ylabel("Number of Patients")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 50. POPULAR NUMERICAL FEATURES
# ============================================================

important_features = [
    "age",
    "bmi",
    "glucose",
    "cholesterol",
    "HbA1c_level",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate"
]

df[important_features].mean().plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Average Patient Health Features")

plt.xlabel("Features")

plt.ylabel("Average Value")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 51. BMI VS GLUCOSE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["bmi"],
    df["glucose"],
    alpha=0.2
)

plt.title("BMI vs Glucose")

plt.xlabel("BMI")

plt.ylabel("Glucose")

plt.tight_layout()

plt.show()


# ============================================================
# 52. AGE VS GLUCOSE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["age"],
    df["glucose"],
    alpha=0.2
)

plt.title("Age vs Glucose")

plt.xlabel("Age")

plt.ylabel("Glucose")

plt.tight_layout()

plt.show()


# ============================================================
# 53. AGE VS CHOLESTEROL
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["age"],
    df["cholesterol"],
    alpha=0.2
)

plt.title("Age vs Cholesterol")

plt.xlabel("Age")

plt.ylabel("Cholesterol")

plt.tight_layout()

plt.show()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==================================================")
print("       DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("==================================================")

print("\nFinal Dataset Shape:")

print(df.shape)

print("\nTotal Patients:", len(df))

print("\nTotal Features:", len(df.columns))

print("\nAll analysis completed!")