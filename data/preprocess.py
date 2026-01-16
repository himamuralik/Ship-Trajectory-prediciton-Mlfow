python
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

