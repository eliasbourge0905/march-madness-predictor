import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_model(y_true, predictions, probabilities):
    """Calculate classification and probability metrics."""
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Log_Loss": log_loss(y_true, probabilities),
        "Brier_Score": brier_score_loss(y_true, probabilities),
    }


def logistic_coefficients(model, feature_names):
    """Return logistic regression coefficients ordered by magnitude."""
    coefficients = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_[0],
        }
    )

    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()

    return coefficients.sort_values(
        "Absolute_Coefficient",
        ascending=False,
    )


def random_forest_importance(model, feature_names):
    """Return random forest feature importances."""
    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    return importance.sort_values(
        "Importance",
        ascending=False,
    )
