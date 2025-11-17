# utils/prediction_utils.py
import os
import glob
import json
import joblib
import pandas as pd
import numpy as np

# Import your data loader utility
from utils import data_utils 

SAVED_MODEL_DIR = "models"

def list_saved_models():
    """Scans the model directory and returns a list of model names."""
    model_list = []
    if os.path.exists(SAVED_MODEL_DIR):
        model_files = glob.glob(os.path.join(SAVED_MODEL_DIR, "*.pkl"))
        model_list = [os.path.basename(f).replace(".pkl", "") for f in model_files]
    return model_list

def get_model_inputs_and_options(model_name, session_filepath):
    """
    Loads model metadata (the .json) and the original dataset
    to build the list of required inputs and their options (for dropdowns).
    """
    
    # 1. Load the features.json file
    features_path = os.path.join(SAVED_MODEL_DIR, f"{model_name}.json")
    try:
        with open(features_path, 'r') as f:
            feature_info = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model metadata (.json): {e}")

    # 2. Load the *original dataset* to get dropdown options
    if not session_filepath:
        raise ValueError("No dataset loaded in session. Please load a dataset on the Modeling page first.")
    
    df = data_utils.load_bigdata(session_filepath)
    if df is None:
        raise ValueError("Failed to load dataset from session.")
    if hasattr(df, "compute"): 
        df = df.compute()
        
    # 3. Build the list of input fields
    inputs = []
    try:
        for col in feature_info["categorical"]:
            unique_vals = df[col].dropna().unique().tolist()
            inputs.append({
                "name": col,
                "type": "select",
                "options": unique_vals[:200] # Limit to 200 options for performance
            })
        for col in feature_info["numeric"]:
            inputs.append({
                "name": col,
                "type": "number",
                "placeholder": f"e.g., {df[col].mean():.2f}"
            })
            
        return sorted(inputs, key=lambda x: x['name'])
        
    except KeyError as e:
        raise KeyError(f"Column mismatch: Feature '{e}' not found in current dataset.")
def make_prediction(model_name, input_data):
    """
    Loads a saved model pipeline and predicts on a single row of data.
    """
    # 1. Define paths
    model_path = os.path.join(SAVED_MODEL_DIR, f"{model_name}.pkl")
    features_path = os.path.join(SAVED_MODEL_DIR, f"{model_name}.json")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(f"Model files for '{model_name}' (.pkl or .json) not found.")
    
    # 2. Load the pipeline (.pkl)
    pipeline = joblib.load(model_path)
    
    # 3. Load the metadata (.json)
    with open(features_path, 'r') as f:
        feature_info = json.load(f)

    # 4. Convert input dict to a single-row DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 5. Make numeric prediction
    prediction = pipeline.predict(input_df)
    
    # 6. Get the raw numeric result (e.g., 0, 1, 2)
    numeric_result = prediction[0]

    # 7. Check if we have a mapping to decode the result
    target_classes = feature_info.get("target_classes")
    
    if target_classes:
        # --- Classification: Decode the number to its label ---
        try:
            # Use the numeric result as an index for the classes list
            final_prediction = target_classes[int(numeric_result)]
        except (IndexError, TypeError):
            # Fallback in case something is wrong
            final_prediction = f"Encoded Value {numeric_result} (Mapping Error)"
    else:
        # --- Regression: Format the number ---
        final_prediction = numeric_result
        if isinstance(final_prediction, (np.int64, np.int32)):
            final_prediction = int(final_prediction)
        if isinstance(final_prediction, (np.float64, np.float32)):
            final_prediction = float(final_prediction)

    return str(final_prediction) # Convert final result to string