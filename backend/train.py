from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN

def train_models(X_train, y_train, task: str):
    if task == "classification":
        model1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model2 = LogisticRegression(max_iter=1000, random_state=42)
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        return model1, model2, "Random Forest", "Logistic Regression"

    elif task == "regression":
        model1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model2 = LinearRegression()
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        return model1, model2, "Random Forest Regressor", "Linear Regression"

    elif task == "clustering":
        model1 = KMeans(n_clusters=4, random_state=42, n_init=10)
        model2 = DBSCAN(eps=0.5, min_samples=5)
        model1.fit(X_train)
        model2.fit(X_train)
        return model1, model2, "K-Means", "DBSCAN"

    else:
        raise ValueError(f"Unknown task: {task}")
