# streamlit_app.py - Deployment Ready Version
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from cdn_optimizer import set_models, vm_placement_with_performance

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="VM Placement & CDN Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Model Training Functions (Integrated)
# =========================
def get_available_models():
    """Return available models based on installed packages"""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "PolynomialRegression": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression())
        ]),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),  # Reduced for speed
        "SVR": SVR(kernel="rbf"),
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    
    return models

def train_models_integrated(df):
    """Integrated model training function"""
    # Prepare data - MODIFY THESE COLUMNS TO MATCH YOUR CSV
    required_columns = ["latency_ms", "egress_cost_per_gb"]
    feature_columns = {
        "numerical": [
            "bandwidth_MBps", "cpu_utilization_%", "ram_utilization_%",
            "request_size_MB", "distance_km", "server_load", "cache_hit_ratio"
        ],
        "categorical": ["storage_tier"]
    }
    
    # Check if required columns exist
    missing_cols = [col for col in required_columns + feature_columns["numerical"] + feature_columns["categorical"] 
                   if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Your CSV should have these columns:")
        st.write("**Target columns:**", required_columns)
        st.write("**Feature columns:**", feature_columns["numerical"] + feature_columns["categorical"])
        return None, None, None
    
    # Prepare features and targets
    X = df[feature_columns["numerical"] + feature_columns["categorical"]]
    y_latency = df["latency_ms"]
    y_egress = df["egress_cost_per_gb"]
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), feature_columns["numerical"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_columns["categorical"])
    ])
    
    # Get available models
    models = get_available_models()
    
    # Results storage
    results = {"latency": {}, "cost": {}}  # Changed egress to cost
    best_models = {"latency": None, "cost": None}  # Changed egress to cost
    best_scores = {"latency": float("inf"), "cost": float("inf")}  # Changed egress to cost
    
    # Training progress
    total_models = len(models) * 2  # For both latency and cost
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_step = 0
    
    # Training for both targets
    for target_name, y in [("latency", y_latency), ("cost", y_egress)]:  # Changed egress to cost
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for name, model in models.items():
            current_step += 1
            progress_bar.progress(current_step / total_models)
            status_text.text(f"Training {name} for {target_name}...")
            
            pipe = Pipeline([
                ("pre", preprocessor),
                ("model", model)
            ])
            
            try:
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                mae = mean_absolute_error(y_test, pred)
                results[target_name][name] = mae
                
                if mae < best_scores[target_name]:
                    best_scores[target_name] = mae
                    best_models[target_name] = pipe
                    
            except Exception as e:
                st.warning(f"Skipped {name} for {target_name}: {str(e)}")
    
    progress_bar.progress(1.0)
    status_text.text("Training complete!")
    
    return results, best_models, best_scores

# =========================
# Load Pre-trained Models (if available)
# =========================
@st.cache_resource
def load_pretrained_models():
    """Load pre-trained models if they exist"""
    models = {}
    model_dir = "models"
    
    if os.path.exists(model_dir):
        latency_path = os.path.join(model_dir, "best_latency.joblib")
        cost_path = os.path.join(model_dir, "best_cost.joblib")
        
        if os.path.exists(latency_path):
            try:
                models["latency"] = joblib.load(latency_path)
            except Exception as e:
                st.warning(f"Could not load latency model: {e}")
        
        if os.path.exists(cost_path):
            try:
                models["cost"] = joblib.load(cost_path)
            except Exception as e:
                st.warning(f"Could not load cost model: {e}")
    
    return models

# Initialize models
if "models" not in st.session_state:
    pretrained = load_pretrained_models()
    if pretrained:
        st.session_state["models"] = pretrained
        st.session_state["using_pretrained"] = True
    else:
        st.session_state["models"] = {}
        st.session_state["using_pretrained"] = False

# =========================
# Sidebar Navigation
# =========================
st.sidebar.markdown("### 🧭 Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["📊 Performance Prediction", "🖥️ VM Placement Optimizer", "📈 Analytics"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 System Status")

active_models = st.session_state.get("models", {})
model_status = {
    "Latency Model": "🟢 Online" if active_models.get("latency") else "🔴 Offline",
    "Cost Model": "🟢 Online" if active_models.get("cost") else "🔴 Offline",
}
for model, status in model_status.items():
    st.sidebar.markdown(f"**{model}:** {status}")
st.sidebar.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

# =========================
# 📊 Performance Prediction
# =========================
if menu == "📊 Performance Prediction":
    st.markdown("## 🔮 ML-Powered Performance Prediction")
    
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            
            st.markdown("### 🔎 Dataset Preview")
            st.dataframe(df.head())
            st.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show column info
            st.markdown("### 📋 Column Information")
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes,
                "Non-Null Count": df.count(),
                "Sample Values": [str(df[col].iloc[0]) if not df[col].isna().all() else "NaN" for col in df.columns]
            })
            st.dataframe(col_info)
            
            # Train models button
            if st.button("🚀 Train Models", type="primary"):
                with st.container():
                    results, best_models, best_scores = train_models_integrated(df)
                    
                    if results and best_models:
                        # Store models in session state
                        st.session_state["models"] = best_models
                        st.session_state["results"] = results
                        st.session_state["best_scores"] = best_scores
                        
                        st.success("✅ Model training completed!")
                        
                        # Show results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ⏱️ Latency Models")
                            latency_df = pd.DataFrame(list(results["latency"].items()), 
                                                    columns=["Model", "MAE (ms)"])
                            latency_df = latency_df.sort_values("MAE (ms)")
                            st.dataframe(latency_df)
                            
                            best_lat_model = latency_df.iloc[0]
                            st.success(f"🏆 Best: {best_lat_model['Model']} (MAE: {best_lat_model['MAE (ms)']:.4f})")
                        
                        with col2:
                            st.markdown("#### 💰 Cost Models")
                            cost_df = pd.DataFrame(list(results["cost"].items()),  # Changed egress to cost
                                                 columns=["Model", "MAE ($/GB)"])
                            cost_df = cost_df.sort_values("MAE ($/GB)")
                            st.dataframe(cost_df)
                            
                            best_cost_model = cost_df.iloc[0]
                            st.success(f"🏆 Best: {best_cost_model['Model']} (MAE: {best_cost_model['MAE ($/GB)']:.4f})")
                        
                        # Visualization
                        fig_lat = go.Figure()
                        fig_lat.add_trace(go.Bar(x=latency_df["Model"], y=latency_df["MAE (ms)"]))
                        fig_lat.update_layout(title="Latency Model Performance", yaxis_title="MAE (ms)")
                        st.plotly_chart(fig_lat, use_container_width=True)
                        
                        fig_cost = go.Figure()
                        fig_cost.add_trace(go.Bar(x=cost_df["Model"], y=cost_df["MAE ($/GB)"], marker_color="orange"))
                        fig_cost.update_layout(title="Cost Model Performance", yaxis_title="MAE ($/GB)")
                        st.plotly_chart(fig_cost, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")

# =========================
# 🖥️ VM Placement Optimizer  
# =========================
elif menu == "🖥️ VM Placement Optimizer":
    st.markdown("## ⚡ VM Placement Optimizer")
    
    # Set models for optimization
    if st.session_state.get("models"):
        set_models(st.session_state["models"])
        if st.session_state.get("using_pretrained"):
            st.success("✅ Using pre-trained models (trained on large dataset)")
        else:
            st.success("✅ Using session-trained models")
    else:
        st.warning("⚠️ No models available. Please train models first or ensure pre-trained models exist.")
    
    st.markdown("## 🎯 Multi-Objective VM Placement with Latency & Cost")
    
    # Performance Parameters
    st.markdown("### ⚙️ Performance Input Parameters")
    with st.form("perf_form"):
        col1, col2 = st.columns(2)
        with col1:
            distance_km = st.number_input("🌍 Distance (km)", 1, 20000, 100)
            bandwidth = st.number_input("📶 Bandwidth (MBps)", 1, 10000, 100)
            server_load = st.slider("🖥️ Server Load (%)", 0, 100, 50) / 100.0
            cache_hit_ratio = st.slider("📂 Cache Hit Ratio (%)", 0, 100, 70) / 100.0
        with col2:
            cpu_util = st.slider("⚡ CPU Utilization (%)", 0, 100, 50)
            ram_util = st.slider("💾 RAM Utilization (%)", 0, 100, 60)
            request_size = st.number_input("📦 Request Size (MB)", 1, 10000, 500)
            storage_tier = st.selectbox("💽 Storage Tier", ["HDD", "SSD", "NVMe"])
        submitted_perf = st.form_submit_button("Save Parameters")
    
    perf_params = {
        "distance_km": distance_km,
        "bandwidth_MBps": bandwidth,
        "server_load": server_load,
        "cache_hit_ratio": cache_hit_ratio,
        "cpu_utilization_%": cpu_util,
        "ram_utilization_%": ram_util,
        "request_size_MB": request_size,
        "storage_tier": storage_tier
    }
    
    # VM/Server Configuration
    st.markdown("### 🖥️ VM & Server Configuration")
    col1, col2 = st.columns(2)
    with col1:
        num_vms = st.number_input("Number of VMs", min_value=2, max_value=20, value=6, step=1)
    with col2:
        num_servers = st.number_input("Number of Servers", min_value=2, max_value=10, value=3, step=1)
    
    vm_names = [f"VM{i+1}" for i in range(num_vms)]
    server_names = [f"Server{j+1}" for j in range(num_servers)]
    
    # VM Demands
    st.markdown("#### VM Resource Demands")
    demands = {}
    cols = st.columns(min(3, len(vm_names)))
    for i, vm in enumerate(vm_names):
        with cols[i % len(cols)]:
            demands[vm] = st.slider(f"{vm} Demand (CPU)", 1, 32, 4, step=1)
    
    # Server Capacities and Costs
    st.markdown("#### Server Capacities & Costs")
    capacities = {}
    server_cost = {}
    cols = st.columns(len(server_names))
    for i, srv in enumerate(server_names):
        with cols[i]:
            capacities[srv] = st.slider(f"{srv} Capacity", 16, 128, 64, step=8)
            server_cost[srv] = st.number_input(f"{srv} Cost ($)", 0.1, 3.0, 1.0, step=0.1)
    
    # Run Optimization
    if st.button("🚀 Run Optimization", type="primary"):
        if not st.session_state.get("models"):
            st.error("Please train models first in the Performance Prediction module!")
        else:
            with st.spinner("Optimizing placement..."):
                result = vm_placement_with_performance(
                    vm_names, server_names, demands, capacities, server_cost, perf_params
                )
            
            if result:
                st.success("✅ Optimization Complete!")
                
                # Results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Latency", f"{result['latency']:.2f} ms")
                with col2:
                    st.metric("Predicted Cost", f"${result['cost']:.4f}/GB")
                
                # VM Assignments
                st.markdown("### 📋 VM Assignments")
                assignment_df = pd.DataFrame([
                    {"VM": vm, "Server": srv, "Demand": demands[vm]}
                    for vm, srv in result["placement"].items()
                ])
                st.dataframe(assignment_df, use_container_width=True)
                
                # Server Metrics
                st.markdown("### 📊 Server Performance")
                server_metrics_df = pd.DataFrame([
                    {"Server": srv, 
                     "Utilization %": metrics["utilization"],
                     "Latency (ms)": metrics["latency"],
                     "Cost ($/GB)": metrics["cost"]}
                    for srv, metrics in result["server_metrics"].items()
                ])
                st.dataframe(server_metrics_df, use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=server_metrics_df["Server"],
                    y=server_metrics_df["Utilization %"],
                    text=server_metrics_df["Utilization %"],
                    textposition="auto"
                ))
                fig.update_layout(title="Server Utilization", yaxis_title="Utilization %")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Optimization failed. Check server capacities and VM demands.")

# =========================
# 📈 Analytics
# =========================
elif menu == "📈 Analytics":
    st.markdown("## 📊 Analytics Dashboard")
    st.info("Advanced analytics features coming soon...")
    
    if "results" in st.session_state:
        st.markdown("### Model Performance History")
        results = st.session_state["results"]
        
        # Create comparison charts separately for latency and cost
        for target in ["latency", "cost"]:  # loop for both metrics
            target_results = results[target]
            
            if target_results:
                results_df = pd.DataFrame({
                    "Model": list(target_results.keys()),
                    "MAE": list(target_results.values())
                })
                
                fig = go.Figure(go.Bar(
                    x=results_df["Model"],
                    y=results_df["MAE"],
                    text=results_df["MAE"].round(3),
                    textposition="auto"
                ))
                
                # Fixed y-axis ranges
                if target == "latency":
                    y_range = [0, 50]
                    y_dtick = 10
                else:  # cost
                    y_range = [0.0, 0.1]
                    y_dtick = 0.02
                
                fig.update_layout(
                    title=f"Model Performance - {target.title()}",
                    yaxis_title="Mean Absolute Error",
                    yaxis=dict(
                        tickformat=".2f" if target == "cost" else ".0f",
                        dtick=y_dtick,
                        range=y_range
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "VM Placement & CDN Optimizer — Deployment Ready Version"
    "</div>",
    unsafe_allow_html=True
)