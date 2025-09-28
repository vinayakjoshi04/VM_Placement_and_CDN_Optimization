import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/vm_placement_cdn_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_validate_data():
    """Load and validate the dataset"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"Available columns: {list(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    
    essential_cols = ["latency_ms", "egress_cost_per_gb"]
    missing_essential = [col for col in essential_cols if col not in df.columns]
    if missing_essential:
        raise ValueError(f"Missing essential target columns: {missing_essential}")
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values")
        print(df.isnull().sum())
    
    return df

def create_preprocessor(available_columns):
    """Create preprocessing pipeline based on available columns"""

    potential_num_cols = [
        "distance_km", "bandwidth_MBps", "server_load", "cache_hit_ratio",
        "cpu_utilization", "ram_utilization", "request_size_MB"
    ]
    potential_cat_cols = ["storage_tier"]
    
    num_cols = [col for col in potential_num_cols if col in available_columns]
    cat_cols = [col for col in potential_cat_cols if col in available_columns]
    
    print(f"Using numerical columns: {num_cols}")
    print(f"Using categorical columns: {cat_cols}")
    
    if not num_cols and not cat_cols:
        raise ValueError("No suitable feature columns found in the dataset")
    
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols))
    
    return ColumnTransformer(transformers), num_cols + cat_cols

def get_models():
    """Define candidate models with better parameters"""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.01, random_state=42, max_iter=2000),
        "PolynomialRegression": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression())
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "SVR": SVR(kernel="rbf", C=1.0, gamma='scale'),
        "XGBoost": XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }

def evaluate_models(X, y, target_name, preprocessor, models):
    """Evaluate all models for a given target"""
    print(f"\n=== Training models for {target_name.upper()} ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    results = {}
    best_model = None
    best_score = float("inf")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name == "PolynomialRegression":

            pipe = Pipeline([
                ("pre", preprocessor),
                ("model", model)
            ])
        else:
            pipe = Pipeline([
                ("pre", preprocessor),
                ("model", model)
            ])
        
        try:
            pipe.fit(X_train, y_train)
            
            y_pred = pipe.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = mae
            
            print(f"  {name}: MAE = {mae:.4f}")
            
            if mae < best_score:
                best_score = mae
                best_model = pipe
                
        except Exception as e:
            print(f"  {name}: FAILED - {str(e)}")
            results[name] = None
    
    return results, best_model, best_score

def main():
    """Main training function"""
    print("Loading dataset...")
    df = load_and_validate_data()
    
    y_latency = df["latency_ms"]
    y_egress = df["egress_cost_per_gb"]
    
    exclude_cols = ["latency_ms", "egress_cost_per_gb"]
    available_feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Available feature columns: {available_feature_cols}")
    
    preprocessor, feature_cols = create_preprocessor(available_feature_cols)
    
    X = df[feature_cols]
    
    print(f"Final feature set: {feature_cols}")
    print(f"Using {len(feature_cols)} features")
    
    models = get_models()
    
    all_results = {"latency": {}, "egress": {}}
    best_models = {"latency": None, "egress": None}
    best_scores = {"latency": float("inf"), "egress": float("inf")}
    
    latency_results, best_latency_model, best_latency_score = evaluate_models(
        X, y_latency, "latency", preprocessor, models
    )
    all_results["latency"] = latency_results
    best_models["latency"] = best_latency_model
    best_scores["latency"] = best_latency_score
    
    egress_results, best_egress_model, best_egress_score = evaluate_models(
        X, y_egress, "egress", preprocessor, models
    )
    all_results["egress"] = egress_results
    best_models["egress"] = best_egress_model
    best_scores["egress"] = best_egress_score
    
    print(f"\n=== Saving Results ===")
    joblib.dump(best_models["latency"], os.path.join(MODEL_DIR, "best_latency.joblib"))
    joblib.dump(best_models["egress"], os.path.join(MODEL_DIR, "best_cost.joblib"))
    
    latency_df = pd.DataFrame([
        {"Model": k, "MAE": v} for k, v in latency_results.items() if v is not None
    ])
    egress_df = pd.DataFrame([
        {"Model": k, "MAE": v} for k, v in egress_results.items() if v is not None
    ])
    
    latency_df.to_csv(os.path.join(MODEL_DIR, "latency_results.csv"), index=False)
    egress_df.to_csv(os.path.join(MODEL_DIR, "cost_results.csv"), index=False)
    
    print("✅ Training complete!")
    print(f"Best latency model: {type(best_models['latency'].named_steps['model']).__name__} (MAE: {best_latency_score:.4f})")
    print(f"Best egress model: {type(best_models['egress'].named_steps['model']).__name__} (MAE: {best_egress_score:.4f})")
    
    return best_models, all_results

if __name__ == "__main__":
    best_models, results = main()