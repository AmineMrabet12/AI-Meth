import pandas as pd


def prepare(df: pd.DataFrame):
    """
    Prepares the input DataFrame for further processing by dropping unnecessary columns,
    filling missing values with the most frequent values, and identifying categorical columns
    for encoding.

    This function performs the following steps:
    1. Drops the `CustomerID` column as it is not needed for analysis.
    2. Prints the most frequent values for specific columns.
    3. Fills missing values in these columns with their respective most frequent (mode) values.
    4. Identifies columns with object data types for one-hot encoding in future steps.

    Args:
        df (pandas.DataFrame): The input DataFrame containing customer data to prepare.

    Returns:
        tuple: A tuple containing:
            - df (pandas.DataFrame): The DataFrame with missing values filled.
            - col_to_encode (list of str): List of column names with object data type for encoding.

    Example:
        >>> df = load_customer_data()
        >>> df_prepared, col_to_encode = prepare(df)

    Notes:
        - Assumes the DataFrame contains columns: 'Tenure', 'WarehouseToHome', 'HourSpendOnApp',
          'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', and 'DaySinceLastOrder'.
        - The function prints the most frequent values for specified columns.
    """

    df = df.drop(columns=['CustomerID'])

    print('--------------------------------------------------------------------')
    print('The most frequent values of Tenure column:', df['Tenure'].mode()[0])
    print('The most frequent values of WarehouseToHome column:', df['WarehouseToHome'].mode()[0])
    print('The most frequent values of HourSpendOnApp column:', df['HourSpendOnApp'].mode()[0])
    print('The most frequent values of OrderAmountHikeFromlastYear column:',
          df['OrderAmountHikeFromlastYear'].mode()[0])
    print('The most frequent values of CouponUsed column:', df['CouponUsed'].mode()[0])
    print('The most frequent values of OrderCount column:', df['OrderCount'].mode()[0])
    print('The most frequent values of DaySinceLastOrder column:', df['DaySinceLastOrder'].mode()[0])
    print('--------------------------------------------------------------------')

    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].mode()[0])
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].mode()[0])
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mode()[0])
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(
        df['OrderAmountHikeFromlastYear'].mode()[0])
    df['CouponUsed'] = df['CouponUsed'].fillna(df['CouponUsed'].mode()[0])
    df['OrderCount'] = df['OrderCount'].fillna(df['OrderCount'].mode()[0])
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].mode()[0])

    col_to_encode = []
    for i in df.columns:
        if df[i].dtype == 'object':
            col_to_encode.append(i)

    return df, col_to_encode
