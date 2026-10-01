from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model on standardized features."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict(model, scaler, X):
    """Return class predictions and Team A win probabilities."""
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    return predictions, probabilities
