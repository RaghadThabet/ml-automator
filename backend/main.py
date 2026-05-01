"""
main.py  

"""

import os, io, sys, traceback
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import run
from train import train_models
from evaluate import evaluate_model

app = FastAPI(title="AutoML Studio API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSION: dict = {}
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR  = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_df(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError(f"Unsupported file type: .{ext}")


def _safe(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    return v


def _clean(d):
    if isinstance(d, dict):  return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, list):  return [_clean(x) for x in d]
    return _safe(d)


# ── POST /api/upload ──────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receive CSV/XLSX, return columns + 5-row preview for the frontend."""
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in {"csv", "xlsx", "xls"}:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Upload .csv or .xlsx.")

    try:
        raw = await file.read()
        df  = _load_df(raw, file.filename)
    except Exception as e:
        raise HTTPException(422, f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(422, "The uploaded file is empty.")

    # Store raw dataframe; wipe any previous training state
    SESSION.clear()
    SESSION["df"]       = df
    SESSION["filename"] = file.filename

    preview = df.head(5).where(pd.notna(df.head(5)), other=None).values.tolist()

    return {
        "filename":     file.filename,
        "rows":         int(len(df)),
        "columns":      df.columns.tolist(),
        "preview_rows": preview,
    }


# ── POST /api/train ───────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    task:   str
    target: str   = ""
    split:  float = 0.8


@app.post("/api/train")
async def train(req: TrainRequest):
    """
    Full AutoML workflow:
      1. Validate inputs
      2. Preprocess + split  (pipeline.run)
      3. Train two models    (train.train_models)
      4. Evaluate & pick best (evaluate.evaluate_model)
      5. Save artefacts
      6. Return results JSON
    """
    # 1 — validate
    if "df" not in SESSION:
        raise HTTPException(400, "No dataset uploaded. Call /api/upload first.")
    if req.task not in {"classification", "regression", "clustering"}:
        raise HTTPException(400, f"Unknown task '{req.task}'.")

    df     = SESSION["df"]
    target = req.target.strip() if req.task != "clustering" else None

    if req.task != "clustering" and not target:
        raise HTTPException(400, "Target column required for classification/regression.")
    if target and target not in df.columns:
        raise HTTPException(400, f"Column '{target}' not found in dataset.")

    # 2 — preprocess
    try:
        X_tr, y_tr, X_te, y_te, pipeline = run(df, target=target, task=req.task)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Preprocessing failed: {e}")

    # 3 — train
    try:
        model1, model2, name1, name2 = train_models(X_tr, y_tr, task=req.task)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Training failed: {e}")

    # 4 — evaluate & compare
    try:
        m1 = evaluate_model(model1, X_te, y_te, task=req.task)
        m2 = evaluate_model(model2, X_te, y_te, task=req.task)

        primary = {"classification": "accuracy", "regression": "r2", "clustering": "silhouette_score"}[req.task]

        s1 = float(m1.get(primary) or 0)
        s2 = float(m2.get(primary) or 0)

        if s1 >= s2:
            best_model, best_name, best_m, other_name, other_s = model1, name1, m1, name2, s2
        else:
            best_model, best_name, best_m, other_name, other_s = model2, name2, m2, name1, s1
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Evaluation failed: {e}")

    # 5 — save artefacts
    try:
        joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_model.pkl"))
        joblib.dump(pipeline,   os.path.join(ARTIFACT_DIR, "preprocess_pipeline.pkl"))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Saving model failed: {e}")

    # 6 — feature importance (optional; never crash for this)
    fi = []
    try:
        raw_fi = None
        if hasattr(best_model, "feature_importances_"):
            raw_fi = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            c = best_model.coef_
            raw_fi = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)

        if raw_fi is not None:
            prep = pipeline.named_steps.get("preprocessing")
            try:
                feat_names = prep.ct.get_feature_names_out().tolist()
            except Exception:
                feat_names = [f"feature_{i}" for i in range(len(raw_fi))]

            paired = sorted(zip(feat_names, raw_fi.tolist()), key=lambda x: x[1], reverse=True)[:10]
            fi = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in paired]
    except Exception:
        fi = []

    # Store in session for /api/model/download
    SESSION.update({"task": req.task, "target": target, "best_model": best_model,
                    "pipeline": pipeline, "metrics": best_m})

    return JSONResponse(_clean({
        "status":      "success",
        "task":        req.task,
        "best_model":  {"name": best_name, "primary_metric": primary, "score": best_m.get(primary)},
        "other_model": {"name": other_name, "score": other_s},
        "metrics":     best_m,
        "feature_importance": fi,
    }))


# ── POST /api/model/download ──────────────────────────────────────────────────

@app.post("/api/model/download")
def download_model():
    """Return the serialised best_model.pkl as a file download."""
    if "best_model" not in SESSION:
        raise HTTPException(400, "No trained model yet. Run /api/train first.")

    path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise HTTPException(404, "Model file not found.")

    return FileResponse(path, media_type="application/octet-stream", filename="best_model.pkl")


# ── GET /api/health ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "session_keys": list(SESSION.keys())}


# ── Run ───────────────────────────────────────────────────────────────────────
# uvicorn main:app --reload --port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
