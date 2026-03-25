import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Loop through all columns
# for col in df.columns:
#     plt.figure()

#     if df[col].dtype == "object":
#         # Categorical -> count plot (histogram equivalent)
#         df[col].value_counts().plot(kind="bar")
#         plt.ylabel("Count")
#     else:
#         # Numerical -> histogram
#         df[col].plot(kind="hist")
#         plt.ylabel("Frequency")

#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.show()

plt.hist(df["age"], bins=10, rwidth=0.8)
plt.title("Distribution of age")
plt.xlabel("age")
plt.ylabel("Frequency")
plt.show()