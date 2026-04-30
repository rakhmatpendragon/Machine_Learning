import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().drop_duplicates()

    # Drop unnecessary columns
    drop_cols = [
        "TransactionID", "AccountID", "DeviceID",
        "IPAddress", "MerchantID", "TransactionDate"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode categorical
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Scale numeric
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
