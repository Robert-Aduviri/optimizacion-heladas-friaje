import re
import numpy as np
import pandas as pd

def fix_headers(df, n_fix_headers=3):
    headers = []
    headers.append(np.array(df.columns))
    for i in range(n_fix_headers):
        headers.append(np.array(df.iloc[i]))
    columns = headers[-1]
    for i in range(n_fix_headers):
        columns = [c1 if c1==c1 else c2 for c1, c2 in zip(columns, headers[n_fix_headers-i-1])]
        df.drop(df.index[0], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.columns = pd.Series(columns).apply(lambda x: re.sub('\n', ' ', x))
    return df  

def check_rare_characters(df, column):
    return df[df[column].apply(lambda x: not re.sub(' ', '', x).isalpha())]

def preprocess_df(df, column):
    df[column] = df[column].apply(lambda x: re.sub('├æ', 'Ñ', x))
    df[column] = df[column].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
    df[column] = df[column].apply(lambda x: re.sub(r'[^a-zA-ZñÑ]+', ' ', x).strip().upper())
    return df