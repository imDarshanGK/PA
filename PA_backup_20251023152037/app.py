import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Use your project title as the Streamlit page title
PROJECT_TITLE = "AI-Based Predictive House Price Estimation using Deep Learning and Ensemble Machine Learning Models"
st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

st.title(PROJECT_TITLE)
st.write("Upload a CSV or place `data/data.csv` in the project folder and press 'Load data'.")

DATA_PATH = Path("data") / "data.csv"

uploaded = st.file_uploader("Upload CSV", type=["csv"]) 

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Loaded uploaded file")
elif DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.success(f"Loaded `{DATA_PATH}`")
else:
    st.info("No dataset provided yet. Upload using the control above or copy your CSV to `data/data.csv`.")
    df = None

if df is not None:
    st.header("Dataset preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head(50))

    st.header("Missing values (top 30)")
    miss = df.isnull().sum().sort_values(ascending=False)
    st.write(miss[miss>0].head(30))

    if "SalePrice" in df.columns:
        st.header("SalePrice stats")
        st.write(df["SalePrice"].describe())
        st.write("Skew:", float(df["SalePrice"].skew()))
        st.write("Log1p skew:", float(np.log1p(df["SalePrice"]).skew()))

    st.header("Quick baseline (numeric features only)")
    if st.button("Run baselines"):
        with st.spinner("Running baselines (LightGBM + MLP). This may take a few minutes)..."):
            try:
                # Lazy imports for heavy libraries so the app can start without them
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import cross_val_score
                from sklearn.neural_network import MLPRegressor
                from sklearn.metrics import mean_squared_error
                import lightgbm as lgb

                y = None
                if "SalePrice" in df.columns:
                    y = df["SalePrice"].copy()
                else:
                    st.error("No `SalePrice` column found — cannot run supervised baseline.")
                X = df.drop(columns=["SalePrice"]) if "SalePrice" in df.columns else df.copy()
                X_num = X.select_dtypes(include=[np.number]).copy()
                if X_num.shape[1] == 0:
                    st.error("No numeric features found to train baselines.")
                else:
                    # Impute
                    imp = SimpleImputer(strategy="median")
                    X_num[:] = imp.fit_transform(X_num)
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X_num)

                    # LightGBM
                    lgbm = lgb.LGBMRegressor(random_state=42, n_estimators=200)
                    scores = cross_val_score(lgbm, Xs, y, cv=5, scoring='neg_mean_squared_error')
                    rmses = np.sqrt(-scores)
                    st.success(f"LightGBM CV RMSE (5-fold): {rmses.mean():.4f} ± {rmses.std():.4f}")

                    # MLP (sklearn)
                    mlp = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=500, random_state=42)
                    scores = cross_val_score(mlp, Xs, y, cv=5, scoring='neg_mean_squared_error')
                    rmses = np.sqrt(-scores)
                    st.success(f"MLP (numeric-only) CV RMSE (5-fold): {rmses.mean():.4f} ± {rmses.std():.4f}")

            except Exception as e:
                st.error(f"Baseline failed: {e}")

    st.markdown("---")
    st.header("Next steps & suggestions")
    st.write("\n- Consider log-transforming `SalePrice` (use np.log1p) when training.\n- Many categorical features exist in Ames (use embeddings or target encoding).\n- Use K-Fold CV and strong baselines (LightGBM).\n- Try TabNet or FT-Transformer for tabular DL approaches.")

else:
    st.stop()
