import pandas as pd

from src.config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)

print("Read CSV Success")
print(df.head)