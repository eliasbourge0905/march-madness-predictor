from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    random_state=42,
):
    """Train a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        oob_score=True,
    )

    model.fit(X_train, y_train)

    return model


def predict(model, X):
    """Return class predictions and Team A win probabilities."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities
