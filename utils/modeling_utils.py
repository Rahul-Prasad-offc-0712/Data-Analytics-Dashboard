# utils/modeling_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# --- NEW IMPORTS REQUIRED FOR SAVING ---
import joblib
import json
import os
# ---------------------------------------

# --- Model Dictionaries ---
REGRESSION_MODELS = {
    "Linear Regression": "sklearn.linear_model.LinearRegression",
    "Random Forest Regressor": "sklearn.ensemble.RandomForestRegressor",
    "Gradient Boosting Regressor": "sklearn.ensemble.GradientBoostingRegressor",
    "XGBoost Regressor": "xgboost.XGBRegressor",
    "SVR": "sklearn.svm.SVR",
    # Add more regressors here
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": "sklearn.linear_model.LogisticRegression",
    "Random Forest Classifier": "sklearn.ensemble.RandomForestClassifier",
    "Gradient Boosting Classifier": "sklearn.ensemble.GradientBoostingClassifier",
    "XGBoost Classifier": "xgboost.XGBClassifier",
    "KNN Classifier": "sklearn.neighbors.KNeighborsClassifier",
    "Decision Tree Classifier": "sklearn.tree.DecisionTreeClassifier",
    # Add more classifiers here
}

def _get_model_instance(model_name, problem_type):
    """Dynamically imports and instantiates a model."""
    try:
        if problem_type == 'regression':
            model_path = REGRESSION_MODELS[model_name]
        else:
            model_path = CLASSIFICATION_MODELS[model_name]
            
        module_name, class_name = model_path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        if "Random" in class_name:
            return model_class(random_state=42)
        else:
            return model_class()
            
    except Exception as e:
        print(f"Error instantiating model {model_name}: {e}")
        return None

def _get_feature_importance_plot(pipeline, feature_names):
    """Generates a feature importance plot for tree-based models."""
    try:
        model = pipeline.named_steps['model']
        
        if not hasattr(model, 'feature_importances_'):
            return None # Not a tree-based model

        # Get feature names from the preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get one-hot encoded feature names
        try:
            ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        except:
             # Fallback for older sklearn
            ohe_features = preprocessor.named_transformers_['cat'].get_feature_names()
        
        # Combine numeric and OHE feature names
        all_feature_names = preprocessor.named_transformers_['num'].feature_names_in_ + list(ohe_features)
        
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(5, len(all_feature_names) // 2)))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([all_feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        # Convert to HTML
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f'<img src="data:image/png;base64,{data}" class="img-fluid" />'
        
    except Exception as e:
        print(f"Feature importance plot error: {e}")
        return f"<p class='text-muted'>Feature importance plot not available for this model type.</p>"


def train_model(df, problem_type, target_col, feature_cols, model_name):
    """
    Trains a model and returns metrics and plots.
    """
    if not target_col or not feature_cols or not model_name or not problem_type:
        return {"error": "Missing inputs. Select problem type, target, features, and model."}

    # --- 1. Data Prep ---
    try:
        # Handle Dask
        if hasattr(df, "compute"):
            df = df.compute()
            
        # Drop rows where target or features are missing
        df_model = df[feature_cols + [target_col]].dropna()
        
        if df_model.empty:
            return {"error": "No valid data after dropping missing values."}

        X = df_model[feature_cols]
        y = df_model[target_col]

    except Exception as e:
        return {"error": f"Data selection error: {e}"}

    # --- 2. Preprocessing Pipeline ---
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    # Numeric pipeline: Impute missing (just in case) and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute missing and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not specified
    )

    # --- 3. Model & Target Prep ---
    model = _get_model_instance(model_name, problem_type)
    if model is None:
        return {"error": f"Could not load model: {model_name}"}

    if problem_type == 'classification':
        # Encode target variable y
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # --- 4. Train/Test Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    except Exception as e:
        return {"error": f"Train/Test split failed: {e}. Not enough data?"}

    # --- 5. Create Full Pipeline & Train ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        return {"error": f"Model training failed: {e}"}

    # --- 6. Evaluate & Return Results ---
    y_pred = pipeline.predict(X_test)
    
    results = {
        "model_name": model_name,
        "problem_type": problem_type,
        "target": target_col,
        "features": ", ".join(feature_cols),
        "metrics": {},
        "plot_html": None
    }
    
    if problem_type == 'regression':
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        results["metrics"]["R-squared"] = f"{r2:.4f}"
        results["metrics"]["RMSE"] = f"{rmse:.4f}"
    else: # Classification
        acc = accuracy_score(y_test, y_pred)
        results["metrics"]["Accuracy"] = f"{acc:.4f}"
        # You could add Confusion Matrix plot here
        
    # Generate feature importance plot if possible
    results["plot_html"] = _get_feature_importance_plot(pipeline, feature_cols)

    return results

# #################################################
# ### --- ADD THIS NEW FUNCTION BELOW --- ###
# #################################################

# --- Define a directory to store models (using your 'models' folder) ---
SAVED_MODEL_DIR = "models"

def save_model_pipeline(df, problem_type, target_col, feature_cols, model_name, save_name):
    """
    Re-trains a model on the FULL dataset and saves it to disk.
    """
    # 1. Create directory
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    
    if not save_name:
        return False, "Model name cannot be empty."

    # 2. Define file paths
    model_path = os.path.join(SAVED_MODEL_DIR, f"{save_name}.pkl")
    features_path = os.path.join(SAVED_MODEL_DIR, f"{save_name}.json")

    if os.path.exists(model_path):
        return False, f"A model with the name '{save_name}' already exists."

    # 3. Re-build and train the pipeline
    try:
        # --- Data Prep ---
        if hasattr(df, "compute"): df = df.compute()
        df_model = df[feature_cols + [target_col]].dropna()
        X = df_model[feature_cols]
        y = df_model[target_col]

        # --- Preprocessing Pipeline ---
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )

        # --- Model & Target Prep ---
        model = _get_model_instance(model_name, problem_type)
        
        # ############ START OF CHANGE ############
        target_classes = None # Default for regression
        if problem_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y)
            # Save the mapping (e.g., [0, 1] -> ["No", "Yes"])
            target_classes = le.classes_.tolist() 
        # ############ END OF CHANGE ############
        
        # --- Create Full Pipeline & Train on ALL data ---
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X, y) 

    except Exception as e:
        return False, f"Failed to re-train model for saving: {e}"

    # 4. Save the pipeline (model) and features (metadata)
    try:
        joblib.dump(pipeline, model_path)
        
        # ############ START OF CHANGE ############
        # Save feature info in a .json file
        feature_info = {
            "features": feature_cols,
            "numeric": numeric_features,
            "categorical": categorical_features,
            "target": target_col,
            "problem_type": problem_type,
            "target_classes": target_classes  # <-- ADDED THIS LINE
        }
        with open(features_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        # ############ END OF CHANGE ############

        return True, f"Model '{save_name}' saved successfully."

    except Exception as e:
        # Clean up if saving fails
        if os.path.exists(model_path): os.remove(model_path)
        if os.path.exists(features_path): os.remove(features_path)
        return False, f"Failed to save model files: {e}"