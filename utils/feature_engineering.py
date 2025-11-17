import os
import time
import shutil
import base64
from io import BytesIO
import numpy as np
import pandas as pd

# Visualization
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.io as pio
except ImportError:
    px = None

# Scaling & Processing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Advanced Features (From your file)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from scipy import stats

# Imbalance Handling
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None


def _backup_file(filepath):
    """Safety: Create a timestamped backup before modifying."""
    try:
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(filepath, os.path.join(backup_dir, f"{os.path.basename(filepath)}.{ts}.bak"))
    except Exception:
        pass


def get_plot_image():
    """Helper: Return base64 image of the current Matplotlib plot."""
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()


def perform_action(filepath, action, column=None, extra=None, sample_threshold=500000, model_dir=None):
    """
    Comprehensive Feature Engineering Dispatcher.
    """
    # Load Data
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    original_shape = df.shape
    msg = ""
    plot_data = None

    # ==========================================
    # 1. ANALYSIS & VISUALIZATION
    # ==========================================
    if action == "Show Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Matrix")
            plot_data = get_plot_image()
            msg = "✅ Correlation Heatmap generated."
        else:
            msg = "⚠️ No numeric columns for heatmap."

    elif action == "Show Missing Map":
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Missing Value Map")
        plot_data = get_plot_image()
        msg = "✅ Missing Value Map generated."

    # ==========================================
    # 2. HANDLING IMBALANCE (Resampling)
    # ==========================================
    elif action in ["SMOTE", "Oversampling", "Undersampling"] and column:
        # 'column' here acts as the Target variable
        target = column
        if target not in df.columns:
            return {"message": "❌ Target column required for resampling."}

        X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target]

        if action == "SMOTE":
            if SMOTE is None: return {"message": "❌ imblearn not installed."}
            sampler = SMOTE(random_state=42)
        elif action == "Oversampling":
            sampler = RandomOverSampler(random_state=42)
        elif action == "Undersampling":
            sampler = RandomUnderSampler(random_state=42)

        # Visualize Class Balance Before
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        y.value_counts().plot(kind='bar', title="Before")

        try:
            X_res, y_res = sampler.fit_resample(X, y)
            # Reconstruct DataFrame
            df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target)], axis=1)

            # Visualize After
            plt.subplot(1, 2, 2)
            df[target].value_counts().plot(kind='bar', title="After", color='orange')
            plt.tight_layout()
            plot_data = get_plot_image()

            msg = f"✅ Applied {action}. New shape: {df.shape}"
        except Exception as e:
            msg = f"❌ Resampling failed: {str(e)}"

    # ==========================================
    # 3. FEATURE SELECTION
    # ==========================================
    elif action in ["SelectKBest (10)", "RFE (Recursive)", "RandomForest Importance"] and column:
        # 'column' acts as Target
        target = column
        X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
        y = df[target]

        if action == "SelectKBest (10)":
            selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
            X_new = selector.fit_transform(X, y)
            cols = X.columns[selector.get_support()]
            df = pd.concat([pd.DataFrame(X_new, columns=cols), y.reset_index(drop=True)], axis=1)
            msg = f"✅ SelectKBest kept: {list(cols)}"

        elif action == "RFE (Recursive)":
            est = LogisticRegression(max_iter=500)
            selector = RFE(est, n_features_to_select=min(10, X.shape[1]))
            selector.fit(X, y)
            cols = X.columns[selector.get_support()]
            df = pd.concat([X[cols].reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            msg = f"✅ RFE kept: {list(cols)}"

        elif action == "RandomForest Importance":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            keep = importances.head(10).index
            df = pd.concat([X[keep].reset_index(drop=True), y.reset_index(drop=True)], axis=1)

            # Plot Importance
            plt.figure(figsize=(8, 5))
            importances.head(10).plot(kind='barh')
            plt.title("Top 10 Features (RF)")
            plot_data = get_plot_image()
            msg = f"✅ Kept Top 10 Features via RF Importance."

    # ==========================================
    # 4. SCALING & TRANSFORMS
    # ==========================================
    elif action in ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"] and column:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            scaler = {
                "Standard Scaling": StandardScaler(),
                "Min-Max Scaling": MinMaxScaler(),
                "Robust Scaling": RobustScaler()
            }[action]

            # Plot Before
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(df[column], kde=True, ax=ax[0], color='gray')
            ax[0].set_title(f"Before")

            df[column] = scaler.fit_transform(df[[column]])

            # Plot After
            sns.histplot(df[column], kde=True, ax=ax[1], color='blue')
            ax[1].set_title(f"After ({action})")
            plt.tight_layout()
            plot_data = get_plot_image()
            msg = f"✅ Applied {action} to '{column}'."

    # ==========================================
    # 5. OUTLIERS (Advanced)
    # ==========================================
    elif action == "Outlier (Isolation Forest)":
        num_df = df.select_dtypes(include=[np.number]).fillna(0)
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(num_df)
        before = len(df)
        df = df[preds == 1]  # Keep inliers
        msg = f"✅ Isolation Forest removed {before - len(df)} rows."

    elif action == "Outlier (Z-Score 3.0)" and column:
        if pd.api.types.is_numeric_dtype(df[column]):
            z = np.abs(stats.zscore(df[column].fillna(0)))
            before = len(df)
            df = df[z < 3]
            msg = f"✅ Z-Score removed {before - len(df)} rows from '{column}'."

    elif action == "Outlier (IQR)" and column:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        before = len(df)
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        msg = f"✅ IQR removed {before - len(df)} rows."

    # ==========================================
    # 6. CLEANING & OTHERS
    # ==========================================
    elif action == "Impute (Mean)":
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].fillna(df[num].mean())
        msg = "✅ Filled numeric missing with Mean."

    elif action == "Impute (Mode)":
        for c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 0)
        msg = "✅ Filled all missing with Mode."

    elif action == "Drop Duplicates":
        df.drop_duplicates(inplace=True)
        msg = "✅ Duplicates removed."

    elif action == "Apply PCA (2 Components)":
        num = df.select_dtypes(include=[np.number]).dropna()
        if num.shape[1] >= 2:
            pca = PCA(n_components=2)
            res = pca.fit_transform(num)
            df['PCA1'] = res[:, 0]
            df['PCA2'] = res[:, 1]
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='PCA1', y='PCA2', data=df)
            plt.title("PCA Result")
            plot_data = get_plot_image()
            msg = "✅ PCA Applied."

    elif action == "Label Encoding" and column:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        msg = f"✅ Label Encoded '{column}'."

    elif action == "One-Hot Encoding" and column:
        df = pd.get_dummies(df, columns=[column])
        msg = f"✅ One-Hot Encoded '{column}'."

    # --- SAVE & RETURN ---
    _backup_file(filepath)
    if filepath.endswith(".parquet"):
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)

    sample_html = df.head(8).to_html(classes='table table-sm table-hover', index=False, border=0)
    return {
        "message": msg,
        "shape_change": f"{original_shape} → {df.shape}",
        "sample_head": sample_html,
        "plot_data": plot_data
    }