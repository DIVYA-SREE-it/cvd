import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_model(data, target_col):
    
    # Encode categorical
    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "F1": f1_score(y_test, preds)
        }

        trained[name] = model

    best_model_name = max(results, key=lambda x: results[x]['F1'])
    best_model = trained[best_model_name]

    return best_model, scaler, X.columns
