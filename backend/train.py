from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import silhouette_score

def train_models(X_train, y_train, task: str):
    if task == "classification":
        model1 = LogisticRegression(max_iter=1000, random_state=42)
        model2 = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
        
        # Both models now receive numeric labels from the pipeline
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        return model1, model2, "Logistic Regression", "XGBoost Classifier", None

    elif task == "regression":
        model1 = LinearRegression()
        model2 = XGBRegressor(random_state=42, n_jobs=-1)
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        return model1, model2, "Linear Regression", "XGBoost Regressor", None

    elif task == "clustering":
        # Find optimal K for KMeans using Silhouette Score
        best_k = 2
        best_score = -1
        max_k = min(10, len(X_train) - 1) if len(X_train) > 2 else 2
        
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_train)
            if len(set(labels)) > 1:
                score = silhouette_score(X_train, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    
        model1 = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        
        # Model robust to outliers
        model2 = DBSCAN(eps=0.5, min_samples=5)
        model1.fit(X_train)
        model2.fit(X_train)
        return model1, model2, f"K-Means (k={best_k})", "DBSCAN", None

    else:
        raise ValueError(f"Unknown task: {task}")
