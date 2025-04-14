def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()
    
    # Optional: normalize manually (z-score standardization)
    df = (df - df.mean()) / df.std()
    
    return df

