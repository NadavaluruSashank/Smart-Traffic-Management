"""
Traffic Congestion Classification
------------------------------------
Trains a classifier to predict traffic state (Free Flow / Moderate /
Congested / Critical) from sensor features.
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from preprocess import load_data, engineer_features, encode_labels, get_feature_columns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "congestion_classifier.pkl")


def train_model(df, feature_cols, target_col="traffic_state_encoded"):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {acc:.2%}\n")
    print("Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Free Flow", "Moderate", "Congested", "Critical"]
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, acc


def get_feature_importance(model, feature_cols):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return importance


if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    df, state_map = encode_labels(df)

    feature_cols = get_feature_columns()
    model, acc = train_model(df, feature_cols)

    print("\nFeature importance:")
    print(get_feature_importance(model, feature_cols))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
