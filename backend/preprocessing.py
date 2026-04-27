import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def feature_filter(df, target=None):

    dropped = []
    n = len(df)
    
    if n == 0:
        return df, []

    for col in df.columns:
        if col == target:
            continue

        s = df[col]
        n_unique = s.nunique()
        valid_count = s.count()

        if n_unique == n:
            dropped.append({"column": col, "reason": "All values unique"})

        elif s.dtype == 'O' and n_unique / n > 0.9:
            dropped.append({
                "column": col,
                "reason": f"High-cardinality text (>{90}% unique values)"
            })

        elif n_unique == 1:
            dropped.append({"column": col, "reason": "Constant column (zero variance)"})

        else:
            missing_ratio = 1 - (valid_count / n)
            max_miss = 0.75 if n < 500 else (0.50 if n < 5000 else 0.30)
            min_valid = max(15, int(n * 0.02))

            if missing_ratio > max_miss:
                dropped.append({
                    "column": col,
                    "reason": (
                        f"Too many missing values "
                        f"({missing_ratio:.0%} missing, threshold {max_miss:.0%})"
                    ),
                })
            elif valid_count < min_valid:
                dropped.append({
                    "column": col,
                    "reason": (
                        f"Too few valid values "
                        f"({valid_count} valid, minimum {min_valid})"
                    ),
                })

    drop_cols = [d["column"] for d in dropped]
    return df.drop(columns=drop_cols, errors="ignore"), dropped

# In High Cardinallity Categrical Target encoder & Frequency encoder are used
# if clustering (y=None) or multi-class classification ----> frequency is used.
# if regression or binary classification ----> TargetEncoder is used.

def _use_frequency_encoding(y):

    if y is None:
        return True                                      
    s = pd.Series(y)
    
    if not pd.api.types.is_numeric_dtype(s):
        return True                                
    
    if pd.api.types.is_integer_dtype(s) and s.nunique() > 2:
        return True                                    

    return False                                      


class _FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, freq_maps=None):
        self.freq_maps = freq_maps  
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            cols = list(self.freq_maps.keys())
            X = pd.DataFrame(X, columns=cols)

        out = np.zeros((len(X), len(self.freq_maps)), dtype=float)
        for i, (col, mapping) in enumerate(self.freq_maps.items()):
            out[:, i] = X[col].map(mapping).fillna(0.0).values
        return out


class Preprocessor:

    def __init__(self):
        self.num_cols      = []
        self.low_cat       = []
        self.high_cat      = []
        self.ct            = None         
        self.encoding_info   = {}       
        self.imputation_info = {}          

    def fit(self, X, y=None):
        
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols      = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self.high_cat = [c for c in cat_cols if X[c].nunique() > 50]
        self.low_cat  = [c for c in cat_cols if X[c].nunique() <= 50]

        if y is not None and hasattr(y, "nunique") and y.nunique() <= 2:
            target_type = "binary"
        else:
            target_type = "continuous"

        transformers = []

        # Numeric pipeline
        if self.num_cols:
            for col in self.num_cols:
                miss = X[col].isna().sum()
                self.imputation_info[col] = f"median ({miss} missing)" if miss > 0 else "none"
                self.encoding_info[col]   = "numeric (RobustScaler)"

            transformers.append(("num", Pipeline([
                ("imp",    SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), self.num_cols))

        # High-cardinality categorical pipeline 

        if self.high_cat:
            for col in self.high_cat:
                miss = X[col].isna().sum()
                self.imputation_info[col] = f"most_frequent ({miss} missing)" if miss > 0 else "none"

            if _use_frequency_encoding(y):
                if y is None:
                    reason = "clustering — no target"
                else:
                    reason = "multi-class target — TargetEncoder not applicable"

                for col in self.high_cat:
                    self.encoding_info[col] = f"frequency encoding ({reason})"

                freq_maps = {
                    col: X[col].value_counts(normalize=True).to_dict()
                    for col in self.high_cat
                }

                transformers.append(("high_cat", Pipeline([
                    ("imp",  SimpleImputer(strategy="most_frequent")),
                    ("freq", _FrequencyEncoder(freq_maps)),
                ]), self.high_cat))

            else:
                for col in self.high_cat:
                    self.encoding_info[col] = (
                        f"target encoding "
                        f"(TargetEncoder, cv=5, smooth=auto, target_type={target_type})"
                    )

                transformers.append(("high_cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("te",  TargetEncoder(
                        cv=5,
                        smooth="auto",
                        target_type=target_type,
                        random_state=42,
                    )),
                ]), self.high_cat))

        #  Low-cardinality categorical pipeline
        if self.low_cat:
            for col in self.low_cat:
                miss = X[col].isna().sum()
                self.imputation_info[col] = f"most_frequent ({miss} missing)" if miss > 0 else "none"
                self.encoding_info[col]   = "one-hot encoding"

            transformers.append(("low_cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.low_cat))

        self.ct = ColumnTransformer(transformers, remainder="drop")
        self.ct.fit(X, y)

        return self

    def transform(self, X):
        return self.ct.transform(X)

    def get_report(self, dropped_cols=None):
        """
        Return a structured dict summarising every column's fate:
        encoding, imputation strategy, and (if provided) why columns were dropped.
        """
        col_reports = []
        all_cols = self.num_cols + self.high_cat + self.low_cat

        for col in all_cols:
            col_reports.append({
                "column":     col,
                "status":     "used",
                "type": (
                    "numeric" if col in self.num_cols else
                    "categorical (high-cardinality)" if col in self.high_cat else
                    "categorical (low-cardinality)"
                ),
                "encoding":   self.encoding_info.get(col, "—"),
                "imputation": self.imputation_info.get(col, "—"),
            })

        dropped_reports = []
        for entry in (dropped_cols or []):
            dropped_reports.append({
                "column": entry["column"] if isinstance(entry, dict) else entry,
                "status": "dropped",
                "reason": entry.get("reason", "filtered out") if isinstance(entry, dict) else "filtered out",
            })

        # Determine which high-card encoder was chosen for the summary block
        high_card_label = (
            "FrequencyEncoder (clustering or multi-class)"
            if self.high_cat and "frequency encoding" in self.encoding_info.get(self.high_cat[0], "")
            else "TargetEncoder (cv=5, smooth=auto)"
        )

        return {
            "summary": {
                "total_features_used":          len(all_cols),
                "total_features_dropped":        len(dropped_reports),
                "numeric_features":              len(self.num_cols),
                "low_cardinality_categorical":   len(self.low_cat),
                "high_cardinality_categorical":  len(self.high_cat),
                "scaling_method":                "RobustScaler (median + IQR)",
                "missing_strategy_numeric":      "median imputation",
                "missing_strategy_categorical":  "most-frequent imputation",
                "high_card_encoding":            high_card_label,
                "low_card_encoding":             "one-hot encoding",
            },
            "columns":         col_reports,
            "dropped_columns": dropped_reports,
            "encoding_info":   self.encoding_info,
            "imputation_info": self.imputation_info,
        }


class PreprocessorWrapper(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.prep = Preprocessor()

    def fit(self, X, y=None):
        self.prep.fit(X, y)
        return self

    def transform(self, X):
        return self.prep.transform(X)

    def get_report(self, dropped_cols=None):
        return self.prep.get_report(dropped_cols=dropped_cols)


    @property
    def ct(self):
        return self.prep.ct

    @property
    def high_cat(self):
        return self.prep.high_cat

    @property
    def low_cat(self):
        return self.prep.low_cat

    @property
    def num_cols(self):
        return self.prep.num_cols
