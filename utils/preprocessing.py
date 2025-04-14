from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column=None):
    df = df.dropna()

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if target_column:
        df_processed = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    else:
        df_processed = X_scaled

    return df_processed
