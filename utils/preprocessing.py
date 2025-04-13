def preprocess_data(df):
    df = df.dropna()
    df = (df - df.mean()) / df.std()
    return df
