# dask_friendly_data_utils.py
import os
import time
import shutil
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import dask.dataframe as dd

import matplotlib.pyplot as plt

# Optional libs - import when needed
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

try:
    import plotly.express as px
except Exception:
    px = None

# sklearn imports kept local inside functions where used
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Feature selection / models imports will be inside functions that need them.

# ----------------- Helpers -----------------
def _ensure_dask(df, npartitions=4):
    """Return a Dask DataFrame (if passed a pandas DF, convert)."""
    if isinstance(df, dd.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        # avoid creating too many partitions for small dfs
        npart = min(npartitions, max(1, (len(df) // 100000) + 1))
        return dd.from_pandas(df, npartitions=npart)
    raise TypeError("Expected pandas.DataFrame or dask.dataframe.DataFrame")

def _ensure_pandas(df):
    """If df is a Dask DataFrame, compute and return pandas DataFrame."""
    if isinstance(df, dd.DataFrame):
        try:
            return df.compute()
        except Exception:
            # fallback: try head -> convert (best-effort)
            return df.head(10)
    return df

def _is_dask(x):
    return isinstance(x, dd.DataFrame) or isinstance(x, dd.Series)

def _safe_head_html(df, n=10):
    try:
        if _is_dask(df):
            pdf = df.head(n)
        else:
            pdf = df.head(n)
        return pdf.to_html(classes="table table-striped table-bordered", index=False)
    except Exception:
        return "<p class='text-muted'>Preview not available.</p>"

def _img_from_matplotlib(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{data}" class="img-fluid" />'

def _backup_file(filepath, backups_dir="backups"):
    try:
        os.makedirs(backups_dir, exist_ok=True)
        basename = os.path.basename(filepath)
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{basename}.backup.{ts}"
        backup_path = os.path.join(backups_dir, backup_name)
        shutil.copy2(filepath, backup_path)
        return backup_path
    except Exception:
        return None

def _save_df(df, filepath):
    """
    Save dataframe (pandas or dask) to csv/parquet and return (True,msg).
    For CSV we try to write a single file (may require compute).
    """
    try:
        # If pandas, wrap to Dask for scalable write
        if isinstance(df, pd.DataFrame):
            ddf = dd.from_pandas(df, npartitions=1)
        elif isinstance(df, dd.DataFrame):
            ddf = df
        else:
            return False, "Unsupported dataframe type for save."

        if filepath.endswith(".parquet"):
            # Dask writes partitioned parquet by default; use single partition if possible
            try:
                # attempt single-file parquet
                ddf.to_parquet(filepath, write_index=False, compute=True)
            except Exception:
                # fallback: compute to pandas then use pandas.to_parquet
                pdf = ddf.compute()
                pdf.to_parquet(filepath, index=False)
        else:
            # CSV: produce a single file. Dask's to_csv may produce multiple files; use single_file option.
            # Use compute if single_file unavailable
            try:
                ddf.to_csv(filepath, single_file=True, index=False)
            except TypeError:
                # older dask versions may not support single_file -> compute and save with pandas
                pdf = ddf.compute()
                pdf.to_csv(filepath, index=False)

        return True, f"Saved to {filepath}"
    except Exception as e:
        return False, f"Save failed: {e}"

def summary_outlier_preview(info):
    """
    Safe HTML summary generator for outlier detection.
    Works even if info is a string OR a dictionary.
    """

    # If info is already an HTML/text string → return it directly
    if isinstance(info, str):
        return f"""
        <div class="alert alert-info" style="font-size:16px;">
            {info}
        </div>
        """

    # Otherwise it's a dict → extract details
    removed = info.get("removed_count", 0)
    method = info.get("method", "").upper()
    cols = info.get("cols", [])
    threshold = info.get("threshold", "")

    html = f"""
    <div class="alert alert-info" style="font-size:16px;">
        <h5>🔍 Outlier Detection Summary</h5>
        <p><b>Method:</b> {method}</p>
        <p><b>Columns:</b> {', '.join(cols) if cols else 'None selected'}</p>
        <p><b>Threshold:</b> {threshold}</p>
        <p><b>Rows Identified as Outliers:</b> {removed}</p>
    </div>
    """

    return html

# ---------- EDA helper ----------
def basic_eda_html(df):
    """
    Return (summary_html, missing_dict, dtypes_dict).
    Accepts pandas or dask DataFrame.
    """
    try:
        ddf = _ensure_dask(df)

        # Describe -> compute (Dask describe doesn't support include="all" reliably)
        try:
            desc_obj = ddf.describe().compute()
            try:
                summary_html = desc_obj.to_html(classes="table table-striped", index=True)
            except Exception:
                summary_html = None
        except Exception:
            summary_html = None

        # missing %
        try:
            missing = (ddf.isna().mean() * 100).compute().round(2)
            missing = pd.Series(missing)
            missing_dict = missing.to_dict()
        except Exception:
            missing_dict = {}

        # dtypes
        try:
            dtypes = ddf.dtypes.compute()
            dtypes = pd.Series(dtypes).astype(str).to_dict()
        except Exception:
            dtypes = {}

        return summary_html, missing_dict, dtypes
    except Exception as e:
        return None, {}, {}

# ---------- OUTLIERS ----------
def preview_remove_outliers(df, cols, method="iqr", iqr_thresh=1.5, z_thresh=3.0, n_estimators=100, contamination=0.01):
    """
    Return (preview_df, info_html). Preview df is pandas DataFrame (computed if needed).
    NOTE: For ML-based methods we compute to pandas.
    """
    try:
        ddf = _ensure_dask(df)
        # ensure columns that exist
        cols_num = [c for c in cols if c in ddf.columns]
        # keep only numeric columns for numeric-based detection
        numeric_cols = []
        # check dtypes by computing small sample
        sample = ddf.head(100)
        for c in cols_num:
            if pd.api.types.is_numeric_dtype(sample[c]):
                numeric_cols.append(c)

        if not numeric_cols:
            return _ensure_pandas(ddf), "<div class='alert alert-warning'>No numeric columns selected for outlier removal.</div>"

        if method == "iqr":
            # compute quantiles per column
            pdf = ddf  # still dask
            mask = None
            for c in numeric_cols:
                try:
                    q1 = pdf[c].quantile(0.25).compute()
                    q3 = pdf[c].quantile(0.75).compute()
                except Exception:
                    # fallback: compute from sample
                    s = pdf[c].dropna().head(10000)
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                iqr = q3 - q1
                low = q1 - iqr_thresh * iqr
                high = q3 + iqr_thresh * iqr
                col_mask = pdf[c].between(low, high)
                if mask is None:
                    mask = col_mask
                else:
                    mask &= col_mask
            try:
                df_clean = pdf[mask].compute()
            except Exception:
                # fallback: compute full pdf then filter
                pdf_comp = pdf.compute()
                mask_pd = ~pd.Series(False, index=pdf_comp.index)
                for c in numeric_cols:
                    Q1 = pdf_comp[c].quantile(0.25)
                    Q3 = pdf_comp[c].quantile(0.75)
                    IQR = Q3 - Q1
                    low = Q1 - iqr_thresh * IQR
                    high = Q3 + iqr_thresh * IQR
                    mask_pd &= pdf_comp[c].between(low, high)
                df_clean = pdf_comp[mask_pd]
            removed = int(len(_ensure_pandas(ddf)) - len(df_clean))
            return df_clean, f"<div class='alert alert-info'>Preview: {removed} rows would be removed (IQR).</div>"

        elif method == "zscore":
            # compute numeric subset to pandas
            pdf = _ensure_pandas(ddf[numeric_cols].fillna(0))
            try:
                from scipy import stats
                zscores = np.abs(stats.zscore(pdf, nan_policy='omit'))
                if zscores.ndim == 1:
                    mask = zscores < z_thresh
                else:
                    mask = np.all(zscores < z_thresh, axis=1)
                df_clean = _ensure_pandas(ddf).loc[mask]
                removed = int(len(_ensure_pandas(ddf)) - len(df_clean))
                return df_clean, f"<div class='alert alert-info'>Preview: {removed} rows would be removed (Z-score).</div>"
            except Exception as e:
                return _ensure_pandas(ddf), f"<div class='alert alert-warning'>Z-score error: {e}</div>"

        elif method == "iforest":
            # IsolationForest requires in-memory data
            try:
                from sklearn.ensemble import IsolationForest
                X = _ensure_pandas(ddf[numeric_cols].fillna(0))
                iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
                preds = iso.fit_predict(X)
                df_all = _ensure_pandas(ddf)
                df_clean = df_all[preds == 1]
                removed = int((preds == -1).sum())
                return df_clean, f"<div class='alert alert-info'>Preview: {removed} rows would be removed (IsolationForest).</div>"
            except Exception as e:
                return _ensure_pandas(ddf), f"<div class='alert alert-warning'>IsolationForest error: {e}</div>"
        else:
            return _ensure_pandas(ddf), "<div class='alert alert-warning'>Unknown outlier method.</div>"

    except Exception as e:
        return _ensure_pandas(df), f"<div class='alert alert-warning'>Outlier preview error: {e}</div>"

def remove_outliers_apply(df, filepath, cols, method="iqr", iqr_thresh=1.5, z_thresh=3.0, n_estimators=100, contamination=0.01, make_backup=True):
    """
    Apply outlier removal and overwrite file. Returns (success_bool, message).
    """
    try:
        ddf = _ensure_dask(df)
        preview_df, info_html = preview_remove_outliers(ddf, cols, method, iqr_thresh, z_thresh, n_estimators, contamination)
        # preview_df is pandas (by design)
        if make_backup and os.path.exists(filepath):
            _backup_file(filepath)
        ok, msg = _save_df(preview_df, filepath)
        if ok:
            return True, f"Outliers removed and saved. {msg}"
        else:
            return False, msg
    except Exception as e:
        return False, f"Remove outliers failed: {e}"

# ---------- IMBALANCE ----------
def preview_resample(df, target, strategy):
    """
    Returns (info_html, plot_html) for preview. Both are strings (html).
    Uses Dask for counting, converts to pandas for sampling/SMOTE previews.
    """
    try:
        ddf = _ensure_dask(df)

        if target not in ddf.columns:
            return "<div class='alert alert-warning'>Select a valid target column.</div>", None

        # count using Dask compute
        vc = ddf[target].value_counts().compute()
        df_before = vc.reset_index()
        df_before.columns = [target, "count"]

        plot_before = None
        if px:
            try:
                plot_before = px.bar(df_before, x=target, y="count", text="count").to_html(full_html=False)
            except Exception:
                plot_before = None

        # Resampling requires pandas (sampling with replacement, SMOTE)
        pdf = _ensure_pandas(ddf)

        if strategy == "oversample":
            maxc = pdf[target].value_counts().max()
            frames = []
            for cls, cnt in pdf[target].value_counts().items():
                subset = pdf[pdf[target] == cls]
                if cnt < maxc:
                    frames.append(subset.sample(maxc, replace=True, random_state=42))
                else:
                    frames.append(subset)
            df_res = pd.concat(frames, ignore_index=True)

        elif strategy == "undersample":
            minc = pdf[target].value_counts().min()
            frames = [pdf[pdf[target] == cls].sample(minc, random_state=42) for cls in pdf[target].value_counts().index]
            df_res = pd.concat(frames, ignore_index=True)

        elif strategy == "smote":
            if SMOTE is None:
                return "<div class='alert alert-warning'>SMOTE not installed (imblearn required).</div>", plot_before

            X = pdf.select_dtypes(include=['number']).fillna(0)
            y = pdf[target]
            if X.shape[1] == 0:
                return "<div class='alert alert-warning'>SMOTE requires numeric features only.</div>", plot_before
            try:
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X, y)
                df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target)], axis=1)
            except Exception as e:
                return f"<div class='alert alert-warning'>SMOTE error: {e}</div>", plot_before
        else:
            return "<div class='alert alert-warning'>Unknown strategy</div>", plot_before

        vc_after = df_res[target].value_counts()
        df_after = vc_after.reset_index()
        df_after.columns = [target, "count"]

        plot_after = None
        if px:
            try:
                plot_after = px.bar(df_after, x=target, y="count", text="count").to_html(full_html=False)
            except Exception:
                plot_after = None

        combined = (
            "<div class='row'>"
            "<div class='col-md-6'><h6>Before</h6>" + (plot_before or df_before.to_html(classes="table table-sm")) + "</div>"
            "<div class='col-md-6'><h6>After (preview)</h6>" + (plot_after or df_after.to_html(classes="table table-sm")) + "</div>"
            "</div>"
        )
        info = f"<div class='alert alert-info'>Before: {vc.to_dict()} | After: {vc_after.to_dict()}</div>"
        return info, combined

    except Exception as e:
        return f"<div class='alert alert-warning'>Resample preview error: {e}</div>", None

def apply_resample(df, filepath, target, strategy, make_backup=True):
    """
    Apply resampling (oversample/undersample/smote) and save the dataset.
    """
    try:
        ddf = _ensure_dask(df)
        if target not in ddf.columns:
            return False, "Invalid target column."

        # convert to pandas for resampling application
        pdf = _ensure_pandas(ddf)

        vc = pdf[target].value_counts()

        if strategy == "oversample":
            maxc = vc.max()
            frames = []
            for cls, cnt in vc.items():
                subset = pdf[pdf[target] == cls]
                if cnt < maxc:
                    frames.append(subset.sample(maxc, replace=True, random_state=42))
                else:
                    frames.append(subset)
            df_res = pd.concat(frames, ignore_index=True)

        elif strategy == "undersample":
            minc = vc.min()
            frames = [pdf[pdf[target] == cls].sample(minc, random_state=42) for cls in vc.index]
            df_res = pd.concat(frames, ignore_index=True)

        elif strategy == "smote":
            if SMOTE is None:
                return False, "SMOTE (imblearn) not installed."

            X = pdf.select_dtypes(include=['number']).fillna(0)
            y = pdf[target]
            if X.shape[1] == 0:
                return False, "SMOTE requires numeric features only."
            try:
                sm = SMOTE(random_state=42)
                X_res, y_res = sm.fit_resample(X, y)
                df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target)], axis=1)
            except Exception as e:
                return False, f"SMOTE error: {e}"
        else:
            return False, "Unknown resampling strategy."

        # backup and save
        if make_backup and os.path.exists(filepath):
            _backup_file(filepath)

        ok, msg = _save_df(df_res, filepath)
        if ok:
            return True, f"Resampling applied and saved successfully. {msg}"
        else:
            return False, msg
    except Exception as e:
        return False, f"Apply resample failed: {e}"

# ---------- TRANSFORMATIONS ----------
def preview_transform(df, cols, transform_type, scaler_type=None, encode_type=None):
    try:
        ddf = _ensure_dask(df)
        if not cols:
            return _ensure_pandas(ddf), "<div class='alert alert-warning'>No columns selected for transform.</div>"

        # Many transformations are easiest in pandas: compute subset
        pdf = _ensure_pandas(ddf)

        if transform_type == "scaler":
            cols_num = [c for c in cols if c in pdf.columns and pd.api.types.is_numeric_dtype(pdf[c])]
            if not cols_num:
                return pdf, "<div class='alert alert-warning'>No numeric columns selected for scaling.</div>"
            if scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            pdf[cols_num] = scaler.fit_transform(pdf[cols_num].fillna(0))
            return pdf, {"message": f"{scaler_type or 'standard'} scaler applied (preview)."}

        if transform_type == "encode":
            if encode_type == "onehot":
                try:
                    df_encoded = pd.get_dummies(pdf, columns=cols, drop_first=False)
                    return df_encoded, {"message": "One-hot encoding (preview)."}
                except Exception as e:
                    return pdf, {"error": f"One-hot preview error: {e}"}
            elif encode_type == "label":
                from sklearn.preprocessing import LabelEncoder
                try:
                    df2 = pdf.copy()
                    le = LabelEncoder()
                    for c in cols:
                        df2[c] = le.fit_transform(df2[c].astype(str))
                    return df2, {"message": "Label encoding (preview)."}
                except Exception as e:
                    return pdf, {"error": f"Label encode preview error: {e}"}
            else:
                return pdf, {"error": "Unknown encode type."}

        if transform_type == "log":
            cols_num = [c for c in cols if c in pdf.columns and pd.api.types.is_numeric_dtype(pdf[c])]
            if not cols_num:
                return pdf, {"error":"No numeric columns selected for log transform."}
            for c in cols_num:
                pdf[c] = np.log1p(pdf[c].fillna(0))
            return pdf, {"message":"Log transform (preview)"}

        return pdf, {"error":"Unknown transform type."}
    except Exception as e:
        return _ensure_pandas(df), {"error": f"Transform preview failed: {e}"}

def apply_transform(df, filepath, cols, transform_type, scaler_type=None, encode_type=None, make_backup=True):
    try:
        ddf = _ensure_dask(df)
        df_new, info = preview_transform(ddf, cols, transform_type, scaler_type, encode_type)
        # df_new is pandas
        if make_backup and os.path.exists(filepath):
            _backup_file(filepath)
        ok, msg = _save_df(df_new, filepath)
        if ok:
            return True, f"Transformation applied and saved. {msg}"
        else:
            return False, msg
    except Exception as e:
        return False, f"Apply transform failed: {e}"

# ---------- FEATURE SELECTION ----------
def preview_feature_selection(df, method, target=None, k=10, estimator_choice="logreg"):
    try:
        ddf = _ensure_dask(df)
        # many feature selection methods require in-memory arrays
        pdf = _ensure_pandas(ddf)

        # ---------- CORRELATION ----------
        if method == "corr":
            num = pdf.select_dtypes(include=['number'])
            if num.shape[1] < 2:
                return "<div class='alert alert-warning'>Not enough numeric columns for correlation.</div>", None
            corr = num.corr()
            if px:
                try:
                    fig = px.imshow(corr, text_auto=True, aspect="auto")
                    return "<div class='alert alert-info'>Correlation (numeric cols)</div>", fig.to_html(full_html=False)
                except Exception:
                    return "<div class='alert alert-info'>Correlation (numeric cols)</div>", corr.to_html(classes="table table-sm")
            return "<div class='alert alert-info'>Correlation (numeric cols)</div>", corr.to_html(classes="table table-sm")

        # ---------- SELECT KBEST ----------
        if method == "kbest":
            if target is None or target not in pdf.columns:
                return "<div class='alert alert-warning'>Select a valid target column for SelectKBest.</div>", None
            X = pdf.select_dtypes(include=['number']).drop(columns=[target], errors='ignore').fillna(0)
            if X.shape[1] == 0:
                return "<div class='alert alert-warning'>No numeric features available for SelectKBest.</div>", None
            y = pdf[target]
            try:
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
                selector.fit(X, y)
                scores = selector.scores_
                feature_names = X.columns
                df_imp = pd.DataFrame({"feature": feature_names, "score": scores}).sort_values("score", ascending=False)
                table = df_imp.head(k).to_html(classes="table table-sm")
                bar = None
                if px:
                    try:
                        bar = px.bar(df_imp.head(k), x="feature", y="score", text="score").to_html(full_html=False)
                    except Exception:
                        bar = None
                return table, bar
            except Exception as e:
                return f"<div class='alert alert-warning'>SelectKBest error: {e}</div>", None

        # ---------- RFE ----------
        if method == "rfe":
            if target is None or target not in pdf.columns:
                return "<div class='alert alert-warning'>Select a valid target column for RFE.</div>", None
            X = pdf.select_dtypes(include=['number']).drop(columns=[target], errors='ignore').fillna(0)
            if X.shape[1] == 0:
                return "<div class='alert alert-warning'>Not enough numeric features for RFE.</div>", None
            y = pdf[target]
            try:
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                est = RandomForestClassifier(n_estimators=100, random_state=42) if estimator_choice == "rf" else LogisticRegression(max_iter=200)
                rfe = RFE(estimator=est, n_features_to_select=min(10, X.shape[1]))
                rfe.fit(X, y)
                ranking = pd.DataFrame({"feature": X.columns, "ranking": rfe.ranking_}).sort_values("ranking")
                table = ranking.to_html(classes="table table-sm")
                return table, None
            except Exception as e:
                return f"<div class='alert alert-warning'>RFE error: {e}</div>", None

        # ---------- RANDOM FOREST IMPORTANCE ----------
        if method == "rf_importance":
            if target is None or target not in pdf.columns:
                return "<div class='alert alert-warning'>Select valid target for RandomForest importance.</div>", None
            X = pdf.select_dtypes(include=['number']).drop(columns=[target], errors='ignore').fillna(0)
            if X.shape[1] == 0:
                return "<div class='alert alert-warning'>No numeric features available for RandomForest.</div>", None
            y = pdf[target]
            try:
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                df_imp = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
                table = df_imp.head(20).to_html(classes="table table-sm")
                bar = None
                if px:
                    try:
                        bar = px.bar(df_imp.head(20), x="feature", y="importance", text="importance").to_html(full_html=False)
                    except Exception:
                        bar = None
                return table, bar
            except Exception as e:
                return f"<div class='alert alert-warning'>RandomForest importance error: {e}</div>", None

        return "<div class='alert alert-warning'>Unknown method</div>", None
    except Exception as e:
        return f"<div class='alert alert-warning'>Feature selection preview error: {e}</div>", None

def apply_feature_selection(df, filepath, target, strategy, k=10, estimator_choice="logreg", make_backup=True): # <-- 1. Added estimator_choice
    """
    Applies feature selection and saves dataset.
    Returns (success_bool, message).
    Strategy: "kbest", "rf_importance", "rfe"
    """
    try:
        ddf = _ensure_dask(df)
        pdf = _ensure_pandas(ddf)

        if target not in pdf.columns:
            return False, "Invalid target column."

        X = pdf.drop(columns=[target])
        y = pdf[target]
        X_num = X.select_dtypes(include=['number']).fillna(0)
        if X_num.shape[1] == 0:
            return False, "No numeric features available for feature selection."

        if strategy == "kbest":
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(score_func=f_classif, k=min(k, X_num.shape[1]))
            X_new = selector.fit_transform(X_num, y)
            selected_cols = X_num.columns[selector.get_support()]
            scores = selector.scores_[selector.get_support()]
            df_imp = pd.DataFrame({"feature": selected_cols, "importance": scores})
            df_res = pd.concat([pd.DataFrame(X_new, columns=selected_cols), y.reset_index(drop=True)], axis=1)

        elif strategy == "rf_importance": # <-- 2. Renamed from "tree"
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
            model.fit(X_num, y)
            importances = model.feature_importances_
            df_imp = pd.DataFrame({"feature": X_num.columns, "importance": importances}).sort_values("importance", ascending=False)
            keep_cols = df_imp.head(k)["feature"].tolist()
            df_res = pd.concat([X_num[keep_cols].reset_index(drop=True), y.reset_index(drop=True)], axis=1)

        elif strategy == "rfe": # <-- 3. Renamed from "l1"
            # --- 4. Replaced L1 logic with correct RFE logic ---
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Use the estimator_choice passed from the route
            if estimator_choice == "rf":
                est = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                est = LogisticRegression(max_iter=200) # Default to logreg
            
            selector = RFE(estimator=est, n_features_to_select=min(k, X_num.shape[1]))
            selector.fit(X_num, y)
            
            selected_cols = X_num.columns[selector.get_support()]
            # Keep only selected columns + target
            df_res = pd.concat([X_num[selected_cols].reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            # --- End of replaced logic ---

        else:
            return False, "Unknown feature selection strategy."

        if make_backup and os.path.exists(filepath):
            _backup_file(filepath)

        ok, msg = _save_df(df_res, filepath)
        if ok:
            return True, f"Feature selection applied & saved. {msg}"
        else:
            return False, msg
    except Exception as e:
        return False, f"apply_feature_selection error: {e}"

# ---------- DIM REDUCTION ----------
def preview_dim_reduction(df, method="pca", n_components=2):
    try:
        ddf = _ensure_dask(df)
        numeric = _ensure_pandas(ddf.select_dtypes(include=['number']).dropna())
        if numeric.shape[1] == 0:
            return numeric, "<div class='alert alert-warning'>No numeric columns for dimensionality reduction.</div>", None
        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_components, numeric.shape[1]))
            transformed = pca.fit_transform(numeric.fillna(0))
            cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
            df_t = pd.DataFrame(transformed, columns=cols)
            expl = pca.explained_variance_ratio_
            fig_html = None
            if px:
                try:
                    fig = px.bar(x=[f"PC{i+1}" for i in range(len(expl))], y=expl, labels={'x':'Component','y':'Explained Variance'})
                    fig_html = fig.to_html(full_html=False)
                except Exception:
                    fig_html = None
            info = f"<div class='alert alert-info'>Explained variance: {expl.tolist()}</div>"
            return df_t, info, fig_html
        else:
            return numeric, "<div class='alert alert-warning'>Method not implemented</div>", None
    except Exception as e:
        return _ensure_pandas(df), f"<div class='alert alert-warning'>Dim reduction preview error: {e}</div>", None

def apply_dim_reduction(df, filepath, method="pca", n_components=2, make_backup=True):
    try:
        ddf = _ensure_dask(df)
        df_t, info, _ = preview_dim_reduction(ddf, method, n_components)
        # df_t is pandas
        if make_backup and os.path.exists(filepath):
            _backup_file(filepath)
        ok, msg = _save_df(df_t, filepath)
        if ok:
            return True, f"Dimensionality reduction applied and saved. {msg}"
        else:
            return False, msg
    except Exception as e:
        return False, f"Apply dim reduction failed: {e}"
