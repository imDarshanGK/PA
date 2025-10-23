import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    xgb = None
    _HAS_XGB = False

st.set_page_config(page_title="AI-Based Predictive House Price Estimation", layout="wide")
st.title("AI-Based Predictive House Price Estimation")
st.markdown("""
This app uses advanced machine learning models (XGBoost and Linear Regression) to estimate house prices based on your input features. The best model is automatically selected for prediction.
""")

# --- Load dataset ---
DATA_PATH = Path("data.csv")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.success(f"Loaded `{DATA_PATH}`")
else:
    st.error("No dataset found. Please add `data.csv` to the project folder.")
    st.stop()

# --- Prepare data ---
target_col = "price"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
exclude_cols = ["date", "street", "city", "statezip", "country"]
df_encoded = df.copy()

# Build explicit category -> code mappings so we can encode inputs the same way
cat_mappings = {}
for col in cat_cols:
    cats = df[col].astype('category').cat.categories.tolist()
    mapping = {c: i for i, c in enumerate(cats)}
    cat_mappings[col] = {
        'mapping': mapping,
        'default_code': mapping.get(df[col].mode().iloc[0]) if not df[col].mode().empty else -1
    }
    # map original df to codes
    df_encoded[col] = df[col].map(mapping).fillna(cat_mappings[col]['default_code']).astype(int)
y = df_encoded[target_col]
X = df_encoded.drop(columns=[target_col])
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

models = {
    "Linear Regression": LinearRegression()
}
if _HAS_XGB:
    models["XGBoost"] = xgb.XGBRegressor(random_state=42, n_estimators=200)
model_rmse = {}
for name, model in models.items():
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, preds))
    model_rmse[name] = rmse
best_model_name = min(model_rmse, key=model_rmse.get)
best_model = models[best_model_name]

# --- Prediction Form ---
st.header("Predict House Price")
input_features = [col for col in X.columns if col not in exclude_cols]
defaults = df[input_features].median(numeric_only=True)
user_input = {}

# Decide numeric vs categorical dynamically from the original dataframe
numeric_cols = df[input_features].select_dtypes(include=[np.number]).columns.tolist()
for col in input_features:
    if col in numeric_cols:
        # check if column is integer-like
        vals = df[col].dropna()
        is_int_like = False
        if not vals.empty:
            is_int_like = np.all(np.mod(vals, 1) == 0)
        if is_int_like:
            user_input[col] = st.number_input(f"{col}", value=int(defaults.get(col, 0)), step=1, format="%d", key=col)
        else:
            user_input[col] = st.number_input(f"{col}", value=float(defaults.get(col, 0)), step=0.01, format="%.2f", key=col)
    else:
        options = df[col].astype(str).unique().tolist()
        user_input[col] = st.selectbox(f"{col}", options, key=col)

if st.button("Predict Price", key="predict_btn"):
    input_df = pd.DataFrame([user_input])
    for col in exclude_cols:
        if col in df.columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                input_df[col] = df[col].median()
            else:
                input_df[col] = df[col].mode()[0]
    # encode categorical columns using the saved mappings
    for col in cat_cols:
        if col in input_df.columns:
            mapping_info = cat_mappings.get(col)
            if mapping_info is not None:
                mapping = mapping_info['mapping']
                default_code = mapping_info['default_code'] if mapping_info.get('default_code') is not None else -1
                input_df[col] = input_df[col].map(mapping).fillna(default_code).astype(int)
            else:
                # fallback
                input_df[col] = input_df[col].astype('category').cat.codes
    input_df = input_df.reindex(columns=X.columns)
    input_imp = imp.transform(input_df)
    input_scaled = scaler.transform(input_imp)
    pred_price = best_model.predict(input_scaled)[0]
    st.markdown("---")
    st.subheader(f"Best Model: {best_model_name} (RMSE: {model_rmse[best_model_name]:,.2f})")
    st.success(f"Predicted House Price: ${pred_price:,.2f}")