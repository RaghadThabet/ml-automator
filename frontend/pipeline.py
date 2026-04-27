import os
import joblib
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from preprocessing import PreprocessorWrapper, feature_filter


def prepare_data(df, task, target=None):
    df = df.drop_duplicates()
    df, _ = feature_filter(df, target)

    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
    else:
        X, y = df, None

    can_stratify = (
        task == "classification"
        and y is not None
        and y.nunique() > 1
        and y.value_counts().min() >= 2
    )

    if y is None:
        X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)
        y_tr, y_te = None, None
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if can_stratify else None,
        )

    return X_tr, X_te, y_tr, y_te


def build_pipeline(y_train, task):
    # SMOTE only applies to classification with imbalanced classes
    use_smote = (
        task == "classification"
        and y_train is not None
        and y_train.value_counts().min() >= 1
        and y_train.value_counts().max() / y_train.value_counts().min() > 4.0
    )
    sampler = SMOTE(random_state=42) if use_smote else "passthrough"

    pipeline = ImbPipeline(steps=[
        ("preprocessing", PreprocessorWrapper()),
        ("smote", sampler),
    ])
    return pipeline


def run_preprocessing_pipeline(X_tr, y_tr, X_te, task):
    pipeline = build_pipeline(y_tr, task=task)
    prep = pipeline.named_steps["preprocessing"]
    smote_step = pipeline.named_steps.get("smote")

    # Step 1: fit+transform preprocessor on train; transform-only on test
    X_tr_processed = prep.fit_transform(X_tr, y_tr)
    X_te_processed = prep.transform(X_te)

    # Step 2: apply SMOTE on already-preprocessed data (classification only)
    y_tr_processed = y_tr
    if task == "classification" and smote_step is not None and smote_step != "passthrough":
        X_tr_processed, y_tr_processed = smote_step.fit_resample(X_tr_processed, y_tr)

    return X_tr_processed, y_tr_processed, X_te_processed, pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")


def save_pipeline(pipeline, name="preprocess_pipeline.pkl"):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, name)
    joblib.dump(pipeline, path)
    print("Saved pipeline at:", path)


def load_pipeline(name="preprocess_pipeline.pkl"):
    path = os.path.join(ARTIFACT_DIR, name)
    return joblib.load(path)


def run(df, target, task):

    X_tr, X_te, y_tr, y_te = prepare_data(df, task=task, target=target)
    X_tr_p, y_tr_p, X_te_p, pipeline = run_preprocessing_pipeline(
        X_tr, y_tr, X_te, task=task
    )
    save_pipeline(pipeline)
    return X_tr_p, y_tr_p, X_te_p, y_te, pipeline
