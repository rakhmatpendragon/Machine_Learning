import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 100

# Generate data
data = {
    "age": np.random.randint(18, 60, n),
    "salary": np.random.randint(3000, 10000, n),
    "gender": np.random.choice(["Male", "Female"], n),
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a more "realistic" target (purchased)
# Rule: higher salary & certain age range -> higher change to purchase
df["purchased"] = np.where(
    (df["salary"] > 6000) & (df["age"] > 25),
    np.random.choice(["Yes", "No"], n, p=[0.7, 0.3]),
    np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
)

# Add missing values
df.loc[np.random.choice(df.index, 10), "salary"] = np.nan

# Add noise to salary
df["salary"] = df["salary"] + np.random.randint(-500, 500, n)

print(df.head())
print(df.info())
print(df.describe())
print(df.describe(include="all"))

plt.hist(df["age"], bins=10, rwidth=0.8)
plt.title("Distribution of age")
plt.xlabel("age")
plt.ylabel("Frequency")
plt.show()

# 1. Check Missing Values & Duplicates (with OUTPUT)
print("=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Duplicates ===")
print(df.duplicated().sum())

# 2. Handle Missing Data (dropna)
df = df.dropna()

# Output check
print("After dropna():")
print(df.isnull().sum())

# 3. Remove Duplicate Data
df = df.drop_duplicates()

# Output check
print("After drop_duplicates():")
print(df.duplicated().sum())

# 4. Drop Unnecessary Columns
columns_to_drop = [
    "TransactionID", "AccountID", "DeviceID",
    "IPAddress", "MerchantID", "TransactionData"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

# Output check
print("Remaining columns:")
print(df.columns)

# 5. Encode Categorical Features (LabelEncodeer)
le = LabelEncoder()

# Apply to all categorical (object) columns
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Output check
print(df.head())