import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Display first 5 rows
print("=== HEAD ===")
print(df.head())

# Display dataset info
print("\n=== INFO ===")
print(df.info())

# Display descriptive statistics
print("\n=== DESCRIBE ===")
print(df.describe())

# HISTOGRAM (DISTRIBUTION OF NUMERICAL DATA)
# Histogram for age
plt.hist(df["age"])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Histogram for salary
plt.hist(df["salary"])
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# CORRELATION MATRIX (HEATMAP)
# Convert categorical to numeric (temporary for correlation)
# df_corr = df.copy()
# df_corr["gender"] = df_corr["gender"].map({"Male": 0, "Female": 1})
# df_corr["purchased"] = df_corr["purchased"].map({"No": 0, "Yes": 1})

# # Correlation heatmap
# corr = df_corr.corr()

# sns.heatmap(corr, annot=True)
# plt.title("Correlation Matrix")
# plt.show()


# Boxplot (Detect Outliers)
# Boxplot for salary
sns.boxplot(x=df["salary"])
plt.title("Salary Boxplot")
plt.show()

# Count Plot (Categoricla Distribution)
# Gender distribution
sns.countplot(x="gender", data=df)
plt.title("Gender Count")
plt.show()

# Purchased distribution
sns.countplot(x="purchased", data=df)
plt.title("Purchased Count")
plt.show()