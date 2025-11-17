import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- Statsmodels imports for Time Series ---
try:
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
    sm = None # Handle missing library
    print("Warning: statsmodels not installed. Time series decomposition and ACF/PACF plots will not work.")


def _img_from_matplotlib(fig):
    """Converts a Matplotlib figure to an HTML img tag."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{data}" class="img-fluid" />'


def _error(msg):
    """Formats an error message as a Bootstrap alert."""
    return f'<div class="alert alert-warning" role="alert">{msg}</div>'


def _ensure_cols_exist(df, cols):
    """Checks if all columns in a list exist in the DataFrame."""
    if not cols:
        return False, "No columns provided."
    for c in cols:
        if c not in df.columns:
            return False, f"Column '{c}' not found."
    return True, None


def _filter_numeric(df, cols):
    """Returns only numeric columns & warns if some were removed."""
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    removed = list(set(cols) - set(numeric_cols))
    return numeric_cols, removed


def generate_plot(df,
                  analysis_type=None,
                  plot_type=None,
                  x_axis=None,
                  y_axis=None,
                  z_axis=None,
                  multi_cols=None,
                  sample_threshold=200000):

    # Normalize plot type
    if plot_type:
        plot_type_norm = plot_type.strip().lower().replace(" ", "_")
    else:
        return _error("No plot type selected.")

    # Normalize analysis type
    if analysis_type:
        analysis_type = analysis_type.strip().lower()
    else:
        return _error("No analysis type selected.")

    # Handle Dask DF or sampling
    try:
        if hasattr(df, "compute"):
            df = df.compute()

        if len(df) > sample_threshold:
            df = df.sample(sample_threshold, random_state=42)
    except:
        pass
    
    # Ensure dataframe is a mutable copy
    df = df.copy()

    # Normalize multi_cols
    if multi_cols and isinstance(multi_cols, str):
        # This handles the case where multi_cols comes from a form as a single string
        multi_cols = [s.strip() for s in multi_cols.split(",") if s.strip()]
    elif multi_cols and isinstance(multi_cols, list):
        # This handles the case where it's already a list (e.g., from request.form.getlist)
        pass


    # -------------------------------------------------------------------
    # ----------------------------- UNIVARIATE ---------------------------
    # -------------------------------------------------------------------
    if analysis_type == "univariate":

        if not x_axis:
            return _error("Select a column for univariate analysis.")

        # Numeric-only plots
        if plot_type_norm in ("hist", "histogram", "kde", "density", "box", "boxplot", "violin"):
            if not pd.api.types.is_numeric_dtype(df[x_axis]):
                return _error(f"'{x_axis}' must be numeric for {plot_type_norm}.")
            
            if plot_type_norm in ("hist", "histogram"):
                fig = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}")
            elif plot_type_norm in ("kde", "density"):
                fig = px.density_contour(df, x=x_axis, title=f"Density Plot of {x_axis}")
            elif plot_type_norm in ("box", "boxplot"):
                fig = px.box(df, y=x_axis, title=f"Box Plot of {x_axis}")
            elif plot_type_norm == "violin":
                fig = px.violin(df, y=x_axis, box=True, points="all", title=f"Violin Plot of {x_axis}")

            return fig.to_html(full_html=False)

        # Categorical plots
        if plot_type_norm in ("count", "countplot", "pie"):
            vc = df[x_axis].value_counts().reset_index()
            vc.columns = [x_axis, "count"]

            if plot_type_norm == "pie":
                fig = px.pie(vc.head(10), names=x_axis, values="count", title=f"Pie Chart of {x_axis} (Top 10)")
            else:
                fig = px.bar(vc.head(25), x=x_axis, y="count", title=f"Count Plot of {x_axis} (Top 25)")

            return fig.to_html(full_html=False)

        return _error("Unsupported univariate plot type.")

    # -------------------------------------------------------------------
    # ----------------------------- BIVARIATE ----------------------------
    # -------------------------------------------------------------------
    if analysis_type == "bivariate":

        if not x_axis or not y_axis:
            # Bubble plot is a special case that needs X, Y, and Z
            if plot_type_norm != "bubble":
                return _error("Select both X and Y columns.")

        # Numeric-only plots
        if plot_type_norm in ("scatter", "line", "hexbin", "regression", "reg", "joint"):

            for col in [x_axis, y_axis]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return _error(f"'{col}' must be numeric for {plot_type_norm}.")

            if plot_type_norm == "scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter: {y_axis} vs {x_axis}")

            elif plot_type_norm == "line":
                df_sorted = df.sort_values(by=x_axis) # Line plots should be sorted
                fig = px.line(df_sorted, x=x_axis, y=y_axis, title=f"Line: {y_axis} vs {x_axis}")

            elif plot_type_norm == "regression" or plot_type_norm == "reg":
                fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"Regression: {y_axis} vs {x_axis}")

            elif plot_type_norm == "hexbin":
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hexbin(df[x_axis], df[y_axis], gridsize=30)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"Hexbin: {y_axis} vs {x_axis}")
                return _img_from_matplotlib(fig)

            elif plot_type_norm == "joint":
                g = sns.jointplot(data=df, x=x_axis, y=y_axis, kind="scatter")
                return _img_from_matplotlib(g.fig)

            return fig.to_html(False)

        # Categorical / mixed
        if plot_type_norm == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar: {y_axis} vs {x_axis}")
            return fig.to_html(False)
        
        # --- BUBBLE PLOT (MOVED FROM MULTIVARIATE) ---
        if plot_type_norm == "bubble":
            if not (x_axis and y_axis and z_axis):
                return _error("Bubble chart requires X, Y, and Z (Size) columns.")

            for col in [x_axis, y_axis, z_axis]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return _error(f"'{col}' must be numeric for bubble chart.")

            fig = px.scatter(df, x=x_axis, y=y_axis, size=z_axis, title=f"Bubble: {y_axis} vs {x_axis}, Sized by {z_axis}")
            return fig.to_html(False)

        return _error("Unsupported bivariate plot type.")

    # -------------------------------------------------------------------
    # ----------------------------- MULTIVARIATE -------------------------
    # -------------------------------------------------------------------
    if analysis_type == "multivariate":

        # 3D scatter uses x, y, z
        if plot_type_norm in ("3d_scatter", "3d"):
            if not (x_axis and y_axis and z_axis):
                return _error("3D Scatter requires X, Y, and Z columns.")
            
            numeric_cols, removed = _filter_numeric(df, [x_axis, y_axis, z_axis])
            if removed:
                return _error(f"All columns for 3D scatter must be numeric. Non-numeric: {removed}")
            
            fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, title=f"3D Scatter: {x_axis}, {y_axis}, {z_axis}")
            return fig.to_html(False)

        # Other plots use multi_cols
        if not multi_cols or len(multi_cols) < 2:
            return _error("Select two or more columns for multivariate analysis.")

        ok, msg = _ensure_cols_exist(df, multi_cols)
        if not ok:
            return _error(msg)

        # Filter numeric columns if needed
        numeric_cols, removed = _filter_numeric(df, multi_cols)

        # Heatmap, pairplot, parallel → numeric only
        if plot_type_norm in ("heatmap", "corr", "correlation_heatmap", "pairplot", "scatter_matrix", "parallel", "parallel_coordinates"):

            if len(numeric_cols) < 2:
                return _error(f"This plot requires at least two numeric columns. Selected numeric: {numeric_cols}")

            if removed:
                return _error(f"Note: The following columns are non-numeric and were excluded: {removed}")

            if plot_type_norm in ("heatmap", "corr", "correlation_heatmap"):
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                return fig.to_html(False)

            if plot_type_norm in ("pairplot", "scatter_matrix"):
                sample = df[numeric_cols].dropna()
                if len(sample) > 2000:
                    sample = sample.sample(2000, random_state=42)
                fig = sns.pairplot(sample)
                fig.fig.suptitle("Pairplot", y=1.02)
                return _img_from_matplotlib(fig.fig)

            if plot_type_norm in ("parallel", "parallel_coordinates"):
                sample = df[numeric_cols].dropna()
                if len(sample) > 5000:
                    sample = sample.sample(5000, random_state=42)
                fig = px.parallel_coordinates(sample, title="Parallel Coordinates")
                return fig.to_html(False)

        return _error("Unsupported multivariate plot type.")

    # -------------------------------------------------------------------
    # ----------------------------- TIME SERIES --------------------------
    # -------------------------------------------------------------------
    if analysis_type == "time_series":
        
        if not x_axis:
            return _error("Select a Date/Time column (X-axis).")
        
        # Ensure x-axis is datetime
        try:
            df[x_axis] = pd.to_datetime(df[x_axis])
        except Exception as e:
            return _error(f"Failed to convert '{x_axis}' to datetime: {e}")
        
        # Sort by time
        df = df.sort_values(by=x_axis)

        # --- Candlestick (special case, uses multi_cols) ---
        if plot_type_norm == "candlestick":
            if not multi_cols or len(multi_cols) != 4:
                return _error("Candlestick plot requires exactly 4 numeric columns (e.g., Open, High, Low, Close).")
            
            ok, msg = _ensure_cols_exist(df, multi_cols)
            if not ok: return _error(msg)
            
            numeric_cols, removed = _filter_numeric(df, multi_cols)
            if removed:
                return _error(f"Columns for Candlestick must be numeric. Non-numeric: {removed}")
            
            o, h, l, c = numeric_cols
            fig = go.Figure(data=[go.Candlestick(x=df[x_axis],
                                               open=df[o],
                                               high=df[h],
                                               low=df[l],
                                               close=df[c])])
            fig.update_layout(title=f"Candlestick Plot ({c} vs {x_axis})")
            return fig.to_html(False)

        # --- All other TS plots require a Y-axis ---
        if not y_axis:
            return _error("Select a Value column (Y-axis).")
        
        if not pd.api.types.is_numeric_dtype(df[y_axis]):
            return _error(f"'{y_axis}' (Value column) must be numeric.")

        # Create the time series as a Series (needed for statsmodels)
        ts = df.set_index(x_axis)[y_axis].dropna()
        
        if ts.empty:
            return _error(f"No valid data found for {y_axis} vs {x_axis}.")

        # --- Time Series Line Plot ---
        if plot_type_norm == "ts_line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"Time Series: {y_axis} vs {x_axis}")
            return fig.to_html(False)

        # --- Rolling Mean ---
        if plot_type_norm == "rolling_mean":
            window = 7 # Hardcode a default window
            rolling = ts.rolling(window=window).mean()
            
            plot_df = pd.DataFrame({
                y_axis: ts,
                f"Rolling Mean (W={window})": rolling
            }).reset_index()

            fig = px.line(plot_df, x=x_axis, y=[y_axis, f"Rolling Mean (W={window})"],
                          title=f"Rolling Mean ({y_axis})")
            return fig.to_html(False)

        # --- Statsmodels plots ---
        if sm is None:
            return _error("Statsmodels library not found. Please install it (`pip install statsmodels`) to use this feature.")
        
        # --- Seasonal Decomposition ---
        if plot_type_norm == "decomposition":
            try:
                # Try to infer a period, default to 12 if data is long enough
                period = 12
                if len(ts) <= 2 * period:
                    period = 7 # Try weekly
                if len(ts) <= 2 * period:
                     period = 1 # No seasonality
                
                res = sm.tsa.seasonal_decompose(ts, model='additive', period=period)
                fig = res.plot() # This is a Matplotlib figure
                fig.set_size_inches(10, 8)
                fig.suptitle(f"Seasonal Decomposition of {y_axis} (Period={period})", y=1.02)
                return _img_from_matplotlib(fig)
            except ValueError as e:
                return _error(f"Decomposition failed: {e}. Try a different column or ensure data has > 2 full periods.")

        # --- ACF/PACF Plot ---
        if plot_type_norm == "acf_pacf":
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                plot_acf(ts, ax=ax1, lags=40)
                ax1.set_title("Autocorrelation (ACF)")
                plot_pacf(ts, ax=ax2, lags=40)
                ax2.set_title("Partial Autocorrelation (PACF)")
                plt.tight_layout()
                return _img_from_matplotlib(fig)
            except Exception as e:
                return _error(f"ACF/PACF plot failed: {e}")

        return _error("Unsupported time series plot type.")

    return _error("Unsupported analysis type.")