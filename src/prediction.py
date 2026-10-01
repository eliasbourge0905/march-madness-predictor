"""Helpers for making predictions from trained models."""


def predict_logistic(model, scaler, matchup_features):
    """Return Team A win prediction and probability from Logistic Regression."""
    scaled = scaler.transform(matchup_features)
    prediction = model.predict(scaled)
    probability = model.predict_proba(scaled)[:, 1]
    return prediction, probability


def predict_random_forest(model, matchup_features):
    """Return Team A win prediction and probability from Random Forest."""
    prediction = model.predict(matchup_features)
    probability = model.predict_proba(matchup_features)[:, 1]
    return prediction, probability
