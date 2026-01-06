import pandas as pd

def map_columns(ml):
    df = ml.copy()

    # Enforce MM/DD/YYYY
    df['Date'] = pd.to_datetime(df['Date'])

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.drop('Date', inplace=True, axis=1)

    df['M/Inns'] = df['M/Inns'].map({
        '1st': 1,
        '2nd': 2,
        'N/A - No Result': 0
    })

    df['Captain'] = df['Captain'].map({
        'Yes': 1,
        'No': 0
    })

    return df
