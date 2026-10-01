"""Reusable model training and evaluation helpers."""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler


def get_features_target(df):
    """Return all *_DIFF model features and the TEAM_A_WIN target."""
    feature_columns = [col for col in df.columns if col.endswith("_DIFF")]
    X = df[feature_columns]
    y = df["TEAM_A_WIN"]
    return X, y


def chronological_split(df, X, y, test_year=2024):
    """Train on seasons before test_year and test only on test_year."""
    train_mask = df["YEAR"] < test_year
    test_mask = df["YEAR"] == test_year

    return (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
    )


def seed_baseline(X_test, y_test):
    """Predict Team A when Team A has the better (lower-numbered) seed."""
    predictions = (X_test["SEED_DIFF"] < 0).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy


def probability_metrics(y_true, probabilities):
    """Calculate ROC-AUC, log loss, and Brier score."""
    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Log_Loss": log_loss(y_true, probabilities),
        "Brier_Score": brier_score_loss(y_true, probabilities),
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Scale features, fit Logistic Regression, and evaluate it."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        **probability_metrics(y_test, probabilities),
    }

    coefficients = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0],
    })
    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values(
        "Absolute_Coefficient", ascending=False
    )

    return model, scaler, predictions, probabilities, metrics, coefficients


def train_random_forest(X_train, y_train, X_test, y_test,
                        n_estimators=100, random_state=42):
    """Fit Random Forest on unscaled features and evaluate it."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        oob_score=True,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        **probability_metrics(y_test, probabilities),
    }

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return model, predictions, probabilities, metrics, feature_importance


def compare_models(seed_accuracy, logistic_metrics, forest_metrics):
    """Create the final model-comparison table."""
    return pd.DataFrame({
        "Model": ["Seed Baseline", "Logistic Regression", "Random Forest"],
        "Accuracy": [
            seed_accuracy,
            logistic_metrics["Accuracy"],
            forest_metrics["Accuracy"],
        ],
        "ROC_AUC": [
            None,
            logistic_metrics["ROC_AUC"],
            forest_metrics["ROC_AUC"],
        ],
        "Log_Loss": [
            None,
            logistic_metrics["Log_Loss"],
            forest_metrics["Log_Loss"],
        ],
        "Brier_Score": [
            None,
            logistic_metrics["Brier_Score"],
            forest_metrics["Brier_Score"],
        ],
    })
