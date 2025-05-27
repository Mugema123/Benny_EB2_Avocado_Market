def preprocess_data(df):
    df = df.drop(columns=['Date'])
    categorical_cols = ['type', 'region']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    X = df.drop(columns=['AveragePrice'])
    y = df['AveragePrice']
    return X, y
