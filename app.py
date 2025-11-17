import os
import time
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file, jsonify
)
from werkzeug.utils import secure_filename

# Utils imports
from utils import data_utils, eda_utils, feature_engineering, modeling_utils, visualization_utils,prediction_utils
from utils import feature_engineering as fe_utils 

# --- Config ---
app = Flask(__name__)
app.secret_key = "replace_with_a_secure_random_string"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_datasets")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
UPLOAD_FOLDER="uploads"

# Maximum rows to load fully; beyond this, Dask sampling kicks in
DEFAULT_SAMPLE_THRESHOLD = 500000

# -----------------------------------------------------------
# -------------------- ROUTES SECTION ------------------------
# -----------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Landing page: dataset upload (CSV / Excel / Parquet / TSV).
    Converts everything internally to Parquet.
    """
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file:
            flash("Please upload a file", "danger")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # ---- Convert any supported file to Parquet ----
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(filepath)
            elif filename.endswith(".tsv"):
                df = pd.read_csv(filepath, sep="\t")
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(filepath)
            else:
                flash("❌ Unsupported file format!", "danger")
                return redirect(url_for("index"))

            # Convert to Parquet
            parquet_name = os.path.splitext(filename)[0] + ".parquet"
            parquet_path = os.path.join(UPLOAD_DIR, parquet_name)
            df.to_parquet(parquet_path, index=False)

            # Save Parquet path to session
            session["filepath"] = parquet_path
            flash(f"✅ File converted and saved as {parquet_name}", "success")
            return redirect(url_for("operation"))

        except Exception as e:
            flash(f"❌ Failed to process file: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/operation")
def operation():
    """
    Operation dashboard after dataset upload.
    """
    if "filepath" not in session:
        flash("Upload a dataset first", "warning")
        return redirect(url_for("index"))
    return render_template("operation.html")

# -----------------------------------------------------------
# -------------------- EDA ROUTE -----------------------------
# -----------------------------------------------------------

@app.route("/eda", methods=["GET", "POST"])
def eda():
    """
    Perform basic and deep EDA, with dataset selection and column drop feature.
    """
    
    # --- 1️⃣ List available datasets ---
    available_files = [
        f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".parquet")
    ]

    selected_file = None
    df = None
    pending_drop_cols = []

    # --- 2️⃣ Check if user selected a dataset ---
    if request.method == "POST":
        selected_file = request.form.get("dataset")
        if selected_file:
            session["filepath"] = os.path.join(UPLOAD_FOLDER, selected_file)
            filepath = session["filepath"]
            df = data_utils.load_bigdata(filepath)  # uses Dask if large

            # Handle Drop Columns feature
            drop_cols = request.form.getlist("drop_cols") or []

            # ---------- Preview Drop ----------
            if "preview_drop" in request.form:
                pending_drop_cols = drop_cols
                flash(f"Selected columns for dropping: {', '.join(drop_cols)}", "info")
                # Create preview table with columns removed (without saving)
                df_preview = df.drop(columns=drop_cols, errors="ignore")
                desc_html, missing_dict, dtypes = eda_utils.basic_eda(df_preview)
                head_html = eda_utils.head_html(df_preview, n=10)
                stats = eda_utils.dashboard_summary(df_preview)
                return render_template(
                    "eda.html",
                    files=available_files,
                    selected_file=selected_file,
                    summary=desc_html,
                    missing=missing_dict,
                    dtypes=dtypes,
                    head=head_html,
                    stats=stats,
                    all_columns=df.columns.tolist(),
                    pending_drop_cols=pending_drop_cols
                )

            # ---------- Update Data ----------
            if "update_data" in request.form:
                if drop_cols:
                    if hasattr(df, "compute"):
                        df = df.compute()
                    df = df.drop(columns=drop_cols, errors="ignore")
                    # Save changes
                    df.to_parquet(filepath, index=False)
                    flash(f"Data updated. Dropped columns: {', '.join(drop_cols)}", "success")
                pending_drop_cols = []

            # ---------- Deep EDA ----------
            target_col = request.form.get("target_col") or None
            if "run_deep_eda" in request.form:
                eda_results = eda_utils.deep_eda_summary(df, target_col=target_col)
                return render_template(
                    "eda.html",
                    files=available_files,
                    selected_file=selected_file,
                    eda=eda_results,
                    all_columns=df.columns.tolist(),
                    pending_drop_cols=pending_drop_cols
                )

            # ---------- Basic EDA ----------
            desc_html, missing_dict, dtypes = eda_utils.basic_eda(df)
            head_html = eda_utils.head_html(df, n=10)
            stats = eda_utils.dashboard_summary(df)
            return render_template(
                "eda.html",
                files=available_files,
                selected_file=selected_file,
                summary=desc_html,
                missing=missing_dict,
                dtypes=dtypes,
                head=head_html,
                stats=stats,
                all_columns=df.columns.tolist(),
                pending_drop_cols=pending_drop_cols
            )

    # --- 3️⃣ Handle GET request ---
    if "filepath" not in session:
        flash("Select a dataset to analyze", "info")
        return render_template("eda.html", files=available_files)

    # If session file exists, load that one
    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)
    desc_html, missing_dict, dtypes = eda_utils.basic_eda(df)
    head_html = eda_utils.head_html(df, n=10)
    stats = eda_utils.dashboard_summary(df)
    selected_file = os.path.basename(filepath)

    return render_template(
        "eda.html",
        files=available_files,
        selected_file=selected_file,
        summary=desc_html,
        missing=missing_dict,
        dtypes=dtypes,
        head=head_html,
        stats=stats,
        all_columns=df.columns.tolist(),
        pending_drop_cols=pending_drop_cols
    )


# -----------------------------------------------------------
# ---------------- FEATURE ENGINEERING -----------------------
# -----------------------------------------------------------
# route
@app.route("/feature-engineering", methods=["GET", "POST"])
def feature_engineering_page():
    """
    Single-page Feature Engineering with tabs:
    - Outliers
    - Data Imbalance
    - Data Transformation
    - Feature Selection (B)
    - Dimensionality Reduction
    """

    # list uploaded dataset files
    available_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".csv", ".parquet"))]

    selected_file = None
    df = None
    preview_html = None
    result_html = None
    numeric_cols = []
    categorical_cols = []
    all_columns = []
    target_options = []

    # helper to load df and compute column lists
    def _prepare_df(filepath):
        df_local = data_utils.load_bigdata(filepath)
        try:
            sample = df_local.compute() if hasattr(df_local, "compute") else df_local
        except Exception:
            sample = df_local
        numeric = sample.select_dtypes(include=["number"]).columns.tolist()
        categorical = sample.select_dtypes(exclude=["number"]).columns.tolist()
        return df_local, sample, numeric, categorical

    if request.method == "POST":
        selected_file = request.form.get("dataset")
        if selected_file:
            session["filepath"] = os.path.join(UPLOAD_FOLDER, selected_file)

        if "filepath" not in session:
            flash("Upload/select a dataset first.", "warning")
            return redirect(url_for("feature_engineering_page"))

        filepath = session["filepath"]
        selected_file = os.path.basename(filepath)
        df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
        all_columns = df_preview_sample.columns.tolist()
        target_options = all_columns.copy()

        section = request.form.get("section")
        action = "preview" if "preview" in request.form else "update" if "update" in request.form else None

        # ---------- OUTLIERS ----------
        if section == "outliers":
            method = request.form.get("outlier_method")
            cols = request.form.getlist("cols_outliers") or []
            iqr_thresh = float(request.form.get("iqr_thresh") or 1.5)
            z_thresh = float(request.form.get("z_thresh") or 3.0)

            if method == "iforest":
                n_estimators = int(request.form.get("if_n", 100))
                contamination = float(request.form.get("if_contamination", 0.01))
            else:
                n_estimators = contamination = None

            if action == "preview":
                preview_df, info = fe_utils.preview_remove_outliers(
                    df=df, cols=cols, method=method,
                    iqr_thresh=iqr_thresh, z_thresh=z_thresh,
                    n_estimators=n_estimators, contamination=contamination
                )
                preview_html = preview_df.head(10).to_html(classes="table table-striped table-bordered", index=False)
                result_html = fe_utils.summary_outlier_preview(info)

            elif action == "update":
                new_df, info = fe_utils.remove_outliers_apply(
                    df=df, filepath=filepath, cols=cols, method=method,
                    iqr_thresh=iqr_thresh, z_thresh=z_thresh,
                    n_estimators=n_estimators, contamination=contamination
                )
                if isinstance(info, dict):
                    removed = info.get("removed_count", 0)
                else:
                    removed = info  # if it's a normal message, keep as-is

                flash(f"Outliers removed: {removed}", "success")
                df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
                preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)
                result_html = fe_utils.summary_outlier_preview(info)

        # ---------- DATA IMBALANCE ----------
        elif section == "imbalance":
            target = request.form.get("imb_target")
            strategy = request.form.get("imb_strategy")

            if action == "preview":
                info, plot_html = fe_utils.preview_resample(df=df, target=target, strategy=strategy)
                result_html = info
                preview_html = plot_html

            elif action == "update":
                success, msg = fe_utils.apply_resample(df=df, filepath=filepath, target=target, strategy=strategy)
                flash(msg, "success" if success else "warning")
                df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
                preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)

        # ---------- DATA TRANSFORMATION ----------
        elif section == "transform":
            trans = request.form.get("transform_type")
            cols = request.form.getlist("transform_cols") or []
            scaler_type = request.form.get("scaler_type")
            encode_type = request.form.get("encode_type")

            if action == "preview":
                preview_df, info = fe_utils.preview_transform(
                    df=df, cols=cols, transform_type=trans,
                    scaler_type=scaler_type, encode_type=encode_type
                )
                preview_html = preview_df.head(10).to_html(classes="table table-striped table-bordered", index=False)
                result_html = info

            elif action == "update":
                new_df, info = fe_utils.apply_transform(
                    df=df, filepath=filepath, cols=cols, transform_type=trans,
                    scaler_type=scaler_type, encode_type=encode_type
                )
                msg = info["message"] if isinstance(info, dict) else str(info)
                flash(msg, "success")
                df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
                preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)

        # ---------- FEATURE SELECTION ----------
        elif section == "selection":
            sel_method = request.form.get("sel_method")
            target = request.form.get("sel_target")
            k = int(request.form.get("k_features") or 10)
            estimator_choice = request.form.get("estimator")

            if action == "preview":
                info_html, plot_html = fe_utils.preview_feature_selection(
                    df=df, method=sel_method, target=target, k=k, estimator_choice=estimator_choice
                )
                result_html = info_html
                preview_html = plot_html

            elif action == "update":
                estimator_choice = request.form.get("estimator") 

                success, message = fe_utils.apply_feature_selection(
                df=df,
                filepath=filepath,
                target=target,strategy=sel_method,k=k,
                                    estimator_choice=estimator_choice  # <-- ADD THIS LINE
                )
                flash(message, "success" if success else "warning")
                df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
                preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)

        # ---------- DIMENSIONALITY REDUCTION ----------
        elif section == "dimred":
            dr_method = request.form.get("dr_method")
            n_comp = int(request.form.get("n_components") or 2)

            if action == "preview":
                transformed_df, info_html, plot_html = fe_utils.preview_dim_reduction(
                    df=df, method=dr_method, n_components=n_comp
                )
                preview_html = transformed_df.head(10).to_html(classes="table table-striped table-bordered", index=False)
                result_html = info_html or plot_html

            elif action == "update":
                applied, message = fe_utils.apply_dim_reduction(
                    df=df, filepath=filepath, method=dr_method, n_components=n_comp
                )
                flash(message, "success" if applied else "warning")
                df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
                preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)

        # DEFAULT: BASIC EDA
        if not preview_html and not result_html:
            desc_html, missing_dict, dtypes = fe_utils.basic_eda_html(df)
            preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)
            result_html = None

    # GET REQUEST
    else:
        if "filepath" in session:
            filepath = session["filepath"]
            selected_file = os.path.basename(filepath)
            df, df_preview_sample, numeric_cols, categorical_cols = _prepare_df(filepath)
            preview_html = df_preview_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)
        else:
            return render_template("feature_engineering.html", files=available_files)

    return render_template(
        "feature_engineering.html",
        files=available_files,
        selected_file=selected_file,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        all_columns=all_columns,
        target_options=target_options,
        preview_html=preview_html,
        result_html=result_html
    )


# -----------------------------------------------------------
# ---------------- VISUALIZATION -----------------------------
# -----------------------------------------------------------
import pandas as pd  # Make sure pandas is imported
import os
from flask import request, session, flash, redirect, url_for, render_template

# ... (other imports and app setup) ...

@app.route("/visualization", methods=["GET", "POST"])
def visualization_page():
    # list files in upload directory
    available_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".csv", ".parquet"))]

    selected_file = None
    df = None
    numeric_cols = []
    categorical_cols = []
    date_time_cols = []
    plot_html = None
    preview_html = None

    if request.method == "POST":
        selected_file = request.form.get("dataset")
        if selected_file:
            session["filepath"] = os.path.join(UPLOAD_FOLDER, selected_file)

        filepath = session.get("filepath")
        if not filepath:
            flash("Please select a dataset first.", "warning")
            return redirect(url_for("visualization_page"))

        # load DF
        df = data_utils.load_bigdata(filepath)

        # ----------------- NEW CHECK (POST) -----------------
        if df is None:
            flash(f"Error: Failed to load or read the file: {os.path.basename(filepath)}. It might be corrupted or in an unsupported format.", "danger")
            session.pop("filepath", None) # Clear the bad filepath
            return redirect(url_for("visualization_page"))
        # ----------------- END CHECK -----------------

        # preview
        if hasattr(df, "compute"):
            try:
                df_preview = df.compute().head(10)
            except Exception:
                df_preview = df.head(10)
        else:
            df_preview = df.head(10)

        preview_html = df_preview.to_html(
            classes="table table-striped table-bordered",
            index=False
        )

        # detect column types
        if hasattr(df, "compute"):
            df_for_types = df.compute()
        else:
            df_for_types = df.copy() # This line is now safe

        # ... (Column detection logic as before) ...
        numeric_cols = df_for_types.select_dtypes(include=["number"]).columns.tolist()
        other_cols = df_for_types.select_dtypes(exclude=["number"]).columns.tolist()
        date_time_cols = []
        categorical_cols = []
        for col in other_cols:
            try:
                converted_col = pd.to_datetime(df_for_types[col], errors='coerce')
                if not converted_col.isnull().all():
                    date_time_cols.append(col)
                else:
                    categorical_cols.append(col)
            except Exception:
                categorical_cols.append(col)
        # ... (End column detection) ...

        # read form inputs
        analysis_type = request.form.get("analysis_type")
        plot_type = request.form.get("chart_type")      
        x = request.form.get("x")
        y = request.form.get("y")
        z = request.form.get("z")
        multi_cols = request.form.getlist("multi_cols") or None

        # generate plot if chart selected
        if plot_type:
            plot_html = visualization_utils.generate_plot(
                df=df,
                analysis_type=analysis_type,
                plot_type=plot_type,
                x_axis=x,
                y_axis=y,
                z_axis=z,
                multi_cols=multi_cols,
                sample_threshold=DEFAULT_SAMPLE_THRESHOLD
            )

        selected_file = os.path.basename(filepath)

    else:  # GET
        if "filepath" in session:
            filepath = session["filepath"]
            selected_file = os.path.basename(filepath)
            df = data_utils.load_bigdata(filepath)

            # ----------------- NEW CHECK (GET) -----------------
            if df is None:
                flash(f"Error: Failed to load file from session: {selected_file}. Please select a new file.", "danger")
                session.pop("filepath", None) # Clear the bad filepath
                return redirect(url_for("visualization_page"))
            # ----------------- END CHECK -----------------

            if hasattr(df, "compute"):
                df_for_types = df.compute()
            else:
                df_for_types = df.copy() # This line is now safe

            preview_html = df_for_types.head(10).to_html(
                classes="table table-striped table-bordered",
                index=False
            )

            # ... (Column detection logic as before) ...
            numeric_cols = df_for_types.select_dtypes(include=["number"]).columns.tolist()
            other_cols = df_for_types.select_dtypes(exclude=["number"]).columns.tolist()
            date_time_cols = []
            categorical_cols = []
            for col in other_cols:
                try:
                    converted_col = pd.to_datetime(df_for_types[col], errors='coerce')
                    if not converted_col.isnull().all():
                        date_time_cols.append(col)
                    else:
                        categorical_cols.append(col)
                except Exception:
                    categorical_cols.append(col)
            # ... (End column detection) ...

    return render_template(
        "visualization.html",
        files=available_files,
        selected_file=selected_file,
        preview_html=preview_html,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        date_time_cols=date_time_cols,
        plot_html=plot_html
    )
# -----------------------------------------------------------
# ---------------- MODELLING SECTION ------------------------
# -----------------------------------------------------------
@app.route("/modeling", methods=["GET", "POST"])
def modeling_page():
    available_files = [
        f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".parquet")
    ]
    # Check if a dataset is in the session
    selected_file = request.form.get("dataset")
    if selected_file:
            session["filepath"] = os.path.join(UPLOAD_FOLDER, selected_file)

    filepath = session.get("filepath")
    if not filepath:
            flash("Please select a dataset first.", "warning")
            return redirect(url_for("visualization_page"))

    
    selected_file = os.path.basename(filepath)
    df = data_utils.load_bigdata(filepath)
    
    if df is None:
        flash(f"Error: Failed to load file {selected_file}. Please re-select.", "danger")
        session.pop("filepath", None)
        return render_template("modeling.html", selected_file=None)

    # --- (Smart column detection logic) ---
    if hasattr(df, "compute"):
        df_for_types = df.compute()
    else:
        df_for_types = df.copy()

    numeric_cols = df_for_types.select_dtypes(include=["number"]).columns.tolist()
    other_cols = df_for_types.select_dtypes(exclude=["number"]).columns.tolist()
    
    date_time_cols = []
    categorical_cols = []

    for col in other_cols:
        try:
            converted_col = pd.to_datetime(df_for_types[col], errors='coerce')
            if not converted_col.isnull().all():
                date_time_cols.append(col)
            else:
                categorical_cols.append(col)
        except Exception:
            categorical_cols.append(col)
            
    # Find numeric columns with low unique values
    low_cardinality_numeric = []
    for col in numeric_cols:
        try:
            unique_count = df_for_types[col].nunique()
            if unique_count <= 25: 
                low_cardinality_numeric.append(col)
        except Exception:
            pass 

    classification_target_cols = sorted(list(set(categorical_cols + low_cardinality_numeric)))
    # --- (End of column logic) ---

    model_results = None
    selected_options = {} 

    if request.method == "POST":
        # #################################################
        # ### START: THIS IS THE FIX ###
        # #################################################

        # Check which button was pressed
        form_action = request.form.get("form_action") 

        # Get common form data
        problem_type = request.form.get("problem_type")
        target_col = request.form.get("target")
        feature_cols = request.form.getlist("features")
        model_name = request.form.get("model_name")
        
        # Store selections to re-populate the form
        selected_options = {
            "problem_type": problem_type,
            "target": target_col,
            "features": feature_cols,
            "model_name": model_name
        }

        # --- Handle "Save Model" action ---
        if form_action == "save_model":
            save_name = request.form.get("save_name")
            
            # Call the save function from modeling_utils
            success, message = modeling_utils.save_model_pipeline(
                df=df_for_types,
                problem_type=problem_type,
                target_col=target_col,
                feature_cols=feature_cols,
                model_name=model_name,
                save_name=save_name
            )
            flash(message, "success" if success else "danger")
            
            # We also re-run the training to keep the results card populated
            model_results = modeling_utils.train_model(
                df=df_for_types,
                problem_type=problem_type,
                target_col=target_col,
                feature_cols=feature_cols,
                model_name=model_name
            )
        
        # --- Handle "Train Model" action ---
        elif form_action == "train":
            model_results = modeling_utils.train_model(
                df=df_for_types,
                problem_type=problem_type,
                target_col=target_col,
                feature_cols=feature_cols,
                model_name=model_name
            )
        
        # ###############################################
        # ### END: THIS IS THE FIX ###
        # ###############################################

    return render_template(
        "modeling.html",
        files=available_files,
        selected_file=selected_file,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        classification_target_cols=classification_target_cols,
        date_time_cols=date_time_cols,
        model_results=model_results,
        selected_options=selected_options
    )
# -----------------------------------------------------------
# ---------------- PREDICTION SECTION -----------------------
# -----------------------------------------------------------

# @app.route("/prediction", methods=["GET", "POST"])
# def prediction_page():
#     """
#     Predict using the last trained model.
#     """
#     if "latest_model_path" not in session:
#         flash("Train a model first under Modelling", "warning")
#         return redirect(url_for("modelling_page"))

#     model_path = session["latest_model_path"]
#     feature_cols = session.get("latest_feature_columns", [])
#     result_table = None

#     if request.method == "POST":
#         file = request.files.get("prediction_file")

#         # Batch prediction from CSV/Parquet
#         if file and file.filename:
#             upload_name = secure_filename(file.filename)
#             pred_path = os.path.join(UPLOAD_DIR, f"pred_input_{int(time.time())}_{upload_name}")
#             file.save(pred_path)

#             out_df = modeling_utils.predict_batch(pred_path, model_path, feature_cols)
#             out_csv = os.path.join(UPLOAD_DIR, f"pred_out_{int(time.time())}.csv")
#             out_df.to_csv(out_csv, index=False)

#             result_table = out_df.head(200).to_html(classes="table table-striped table-sm", index=False)
#             session["last_prediction_file"] = out_csv

#         # Single-row manual prediction
#         else:
#             input_data = {}
#             for c in feature_cols:
#                 val = request.form.get(f"feature__{c}")
#                 if val == "" or val is None:
#                     input_data[c] = None
#                 else:
#                     try:
#                         input_data[c] = float(val)
#                     except ValueError:
#                         input_data[c] = val

#             pred = modeling_utils.predict_single(input_data, model_path, feature_cols)
#             result_table = f"<table class='table table-sm'><tr><th>Prediction</th></tr><tr><td>{pred}</td></tr></table>"

#     return render_template("prediction.html", cols=feature_cols, result_table=result_table)


# @app.route("/download_predictions")
# def download_predictions():
#     """
#     Download prediction output CSV.
#     """
#     f = session.get("last_prediction_file")
#     if not f or not os.path.exists(f):
#         flash("No prediction output file available", "warning")
#         return redirect(url_for("prediction_page"))
#     return send_file(f, as_attachment=True)

@app.route("/prediction", methods=["GET"])
def prediction_page():
    # Get model list from the new util
    model_list = prediction_utils.list_saved_models()
    return render_template("prediction.html", model_list=model_list)


@app.route("/get_model_inputs/<model_name>", methods=["GET"])
def get_model_inputs(model_name):
    # This API populates the dynamic form
    try:
        # Get the path to the dataset from the session
        filepath = session.get("filepath")
        
        # Call the new util
        inputs = prediction_utils.get_model_inputs_and_options(model_name, filepath)
        
        return jsonify({"inputs": inputs})
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route("/predict/<model_name>", methods=["POST"])
def do_prediction(model_name):
    # This API runs the prediction
    try:
        input_data = request.json
        
        # Call the new util
        result_str = prediction_utils.make_prediction(model_name, input_data)
        
        return jsonify({"prediction": result_str})
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# ---------------- DASHBOARD METRICS ------------------------
# -----------------------------------------------------------

@app.route("/dashboard")
def dashboard_page():
    """
    Lightweight dashboard of key dataset metrics.
    """
    if "filepath" not in session:
        flash("Upload dataset first", "warning")
        return redirect(url_for("index"))

    filepath = session["filepath"]
    df = data_utils.load_bigdata(filepath)
    stats = eda_utils.dashboard_summary(df)
    return render_template("dashboard.html", stats=stats)


# -----------------------------------------------------------
# ---------------- HEALTH CHECK -----------------------------
# -----------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# -----------------------------------------------------------
# ---------------- RUN SERVER -------------------------------
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
