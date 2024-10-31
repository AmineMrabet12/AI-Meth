import pandas as pd


def prepare(df: pd.DataFrame):

    df = df.drop(columns=['CustomerID'])
    
    print('--------------------------------------------------------------------')
    print('The most frequent values of Tenure column:', df['Tenure'].mode()[0])
    print('The most frequent values of WarehouseToHome column:', df['WarehouseToHome'].mode()[0])
    print('The most frequent values of HourSpendOnApp column:', df['HourSpendOnApp'].mode()[0])
    print('The most frequent values of OrderAmountHikeFromlastYear column:', df['OrderAmountHikeFromlastYear'].mode()[0])
    print('The most frequent values of CouponUsed column:', df['CouponUsed'].mode()[0])
    print('The most frequent values of OrderCount column:', df['OrderCount'].mode()[0])
    print('The most frequent values of DaySinceLastOrder column:', df['DaySinceLastOrder'].mode()[0])
    print('--------------------------------------------------------------------')

    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].mode()[0])
    df['WarehouseToHome'] = df['WarehouseToHome'].fillna(df['WarehouseToHome'].mode()[0])
    df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mode()[0])
    df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].mode()[0])
    df['CouponUsed'] = df['CouponUsed'].fillna(df['CouponUsed'].mode()[0])
    df['OrderCount'] = df['OrderCount'].fillna(df['OrderCount'].mode()[0])
    df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].mode()[0])

    col_to_encode = []
    for i in df.columns:
        if df[i].dtype == 'object':
            col_to_encode.append(i)

    return df, col_to_encode