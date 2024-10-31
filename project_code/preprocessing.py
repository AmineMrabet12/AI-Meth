from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd

def preprocess(df, col_to_encode):
    """
    Preprocess the input DataFrame by applying one-hot encoding to specified columns,
    scaling features, and splitting the data into training and test sets.

    This function performs the following preprocessing steps:
    1. One-hot encodes the specified categorical columns.
    2. Splits the data into features (X) and target (y).
    3. Splits X and y into training and test sets.
    4. Scales the features using standard scaling.
    5. Saves the encoder and scaler objects for future use.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset to preprocess.
        col_to_encode (list of str): List of column names to be one-hot encoded.

    Returns:
        tuple: A tuple containing:
            - X_train (numpy.ndarray): The scaled training data features.
            - X_test (numpy.ndarray): The scaled test data features.
            - y_train (pandas.Series): The training data target labels.
            - y_test (pandas.Series): The test data target labels.

    Saves:
        - Encoder as `OneHotEncoder.joblib` in the `models` directory.
        - Scaler as `StandardScaler.joblib` in the `models` directory.

    Example:
        >>> df = load_data()
        >>> col_to_encode = ['PreferredLoginDevice', 'PreferredPaymentMode']
        >>> X_train, X_test, y_train, y_test = preprocess(df, col_to_encode)

    Notes:
        - The target column `Churn` is assumed to be present in the input DataFrame.
        - The function uses a test size of 20% and a random state of 42 for reproducibility.
    """

    encoder = OneHotEncoder(sparse_output=False, drop='first')
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

    dump(encoder, '../models/OneHotEncoder.joblib')
    dump(encoder, '../models/StandardScaler.joblib')

    return X_train, X_test, y_train, y_test