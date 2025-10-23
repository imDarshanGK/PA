import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except Exception:
    lgb = None
    _HAS_LIGHTGBM = False

def analyze_dataset(file_path):
    print("\n=== Ames Housing Dataset Analysis ===\n")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        
        # Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            print("\nColumns with missing values:")
            print(missing)
        
        # SalePrice analysis if exists
        if 'SalePrice' in df.columns:
            print("\nSalePrice Statistics:")
            print(df['SalePrice'].describe())
            print(f"Skewness: {df['SalePrice'].skew():.3f}")
            print(f"Log1p Skewness: {np.log1p(df['SalePrice']).skew():.3f}")
            
            # Create histogram
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            df['SalePrice'].hist()
            plt.title('SalePrice Distribution')
            
            plt.subplot(1, 2, 2)
            np.log1p(df['SalePrice']).hist()
            plt.title('Log(SalePrice) Distribution')
            plt.tight_layout()
            plt.savefig('price_distribution.png')
            print("\nSaved price distribution plots to 'price_distribution.png'")
            
            # Quick baseline
            print("\nTraining baseline models...")
            X = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice'])
            y = df['SalePrice']
            
            # Impute missing values
            imp = SimpleImputer(strategy='median')
            X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
            
            # Scale features
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # LightGBM baseline (only if available)
            if _HAS_LIGHTGBM:
                try:
                    lgb_model = lgb.LGBMRegressor(random_state=42)
                    scores = cross_val_score(lgb_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
                    rmse_scores = -scores
                    print(f"\nLightGBM 5-fold CV RMSE: {rmse_scores.mean():.0f} ± {rmse_scores.std():.0f}")

                    # Feature importance
                    lgb_model.fit(X, y)
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': lgb_model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)

                    print("\nTop 10 important features:")
                    print(importance)

                    # Save feature importance plot
                    plt.figure(figsize=(10, 6))
                    plt.bar(importance['feature'], importance['importance'])
                    plt.xticks(rotation=45, ha='right')
                    plt.title('Top 10 Important Features')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png')
                    print("\nSaved feature importance plot to 'feature_importance.png'")
                except Exception as e:
                    print(f"LightGBM step failed: {e}")
            else:
                print("\nLightGBM not installed; skipping LightGBM baseline and feature importance.")
            
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    data_path = Path("data") / "data.csv"
    if data_path.exists():
        analyze_dataset(data_path)
    else:
        print(f"Please place your dataset at {data_path}")