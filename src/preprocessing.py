import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_clean(path):
    df = pd.read_csv(path)

    df.replace('?', np.nan, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy='mean')
        df[num_cols] = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols
        )

    return df
