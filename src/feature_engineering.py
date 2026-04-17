import pandas as pd

def feature_engineering(df):
    df = df.copy()

    if 'weight' in df.columns and 'height' in df.columns:
        df['BMI'] = df['weight'] / ((df['height']/100)**2)

    # Safe column handling
    for col in ['steps', 'sleep_hours', 'avg_hr']:
        if col not in df.columns:
            df[col] = df[col].median() if col in df else 0

    df['lifestyle_score'] = (
        df['steps']*0.3 +
        df['sleep_hours']*0.3 +
        (1/(df['avg_hr']+1))*0.4
    )

    df['stress_index'] = df['avg_hr'] / (df['sleep_hours']+1)

    return df
