from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd

def preprocess(df, col_to_encode):

    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = encoder.fit_transform(df[col_to_encode])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(col_to_encode))
    df_encoded = pd.concat([df.drop(columns=col_to_encode), encoded_df], axis=1)

    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=.2, random_state=42)
    scaler = StandardScaler()
    
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    dump(encoder, 'models/OneHotEncoder.joblib')
    dump(encoder, 'models/StandardScaler.joblib')

    return X_train, X_test, y_train, y_test