from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
import numpy as np

def evaluate_model(model, X_test, y_test, task: str) -> dict:
    if task == "classification":
        y_pred = model.predict(X_test)
        return {
            "accuracy":         accuracy_score(y_test, y_pred),
            "precision":        precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":           recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1":               f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    elif task == "regression":
        y_pred = model.predict(X_test)
        return {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "r2":  r2_score(y_test, y_pred),
        }

    elif task == "clustering":
        labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_test)
        score = silhouette_score(X_test, labels) if len(set(labels)) > 1 else 0.0
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        inertia = float(model.inertia_) if hasattr(model, "inertia_") else None
        return {
            "silhouette_score": score,
            "n_clusters":       n_clusters,
            "inertia":          inertia,
        }

    else:
        raise ValueError(f"Unknown task: {task}")
