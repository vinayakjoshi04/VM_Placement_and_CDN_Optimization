# 🚀 VM Placement & CDN Optimizer

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

> An intelligent machine learning-powered system for optimizing virtual machine placement and predicting CDN performance. Combines advanced optimization algorithms with predictive models to minimize latency and costs in distributed computing environments.

**[🌐 Live Demo](https://vmplacementandcdnoptimization-mnzkzoya3ysevnq3h3qsj6.streamlit.app/)** | **[📝 Report Bug](https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization/issues)** | **[✨ Request Feature](https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization/issues)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The VM Placement & CDN Optimizer addresses the complex challenge of efficiently distributing virtual machines across servers while optimizing Content Delivery Network performance. Using machine learning and mathematical optimization, it helps organizations minimize latency, reduce costs, and maximize resource utilization.

### Problem Statement

Organizations face challenges in:
- ✅ Efficiently distributing workloads across servers
- ✅ Minimizing network latency for end users
- ✅ Reducing operational costs while maintaining performance
- ✅ Managing capacity constraints and resource allocation

### Solution

This tool provides:
- **Predictive Analytics**: ML models forecast latency and costs based on system parameters
- **Intelligent Optimization**: Hybrid algorithms find optimal VM-to-server assignments
- **Real-Time Insights**: Interactive dashboard for monitoring and decision-making

---

## ✨ Key Features

### 🔮 Performance Prediction Engine

Train and compare **8 different ML algorithms**:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression (degree 2)
- Random Forest Regressor
- Support Vector Regression (SVR)
- XGBoost
- LightGBM

**Automatic Model Selection**: System automatically selects best-performing models based on Mean Absolute Error (MAE)

### ⚡ VM Placement Optimizer

**Optimization Techniques**:
- **Integer Linear Programming (ILP)**: Guarantees optimal solutions using PuLP solver
- **Greedy Fallback Algorithm**: Fast approximate solutions for large-scale problems
- **Multi-Constraint Handling**: Respects server capacity, VM demands, and cost constraints

**Configurable Parameters**:
- Distance (km)
- Bandwidth (MBps)
- Server Load (%)
- Cache Hit Ratio (%)
- CPU/RAM Utilization (%)
- Request Size (MB)
- Storage Tier (HDD/SSD/NVMe)

### 📊 Interactive Analytics Dashboard

- Model performance comparison visualizations
- Server utilization monitoring
- Historical performance tracking
- Real-time optimization results
- Resource allocation insights

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit, Plotly |
| **ML Framework** | scikit-learn, XGBoost, LightGBM |
| **Optimization** | PuLP (CBC Solver) |
| **Data Processing** | Pandas, NumPy |
| **Model Persistence** | Joblib |
| **Deployment** | Streamlit Cloud, Docker |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization.git
cd VM_Placement_and_CDN_Optimization

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

The application will automatically open in your browser at `http://localhost:8501`

### Alternative Installation Methods

<details>
<summary><b>Using Conda</b></summary>

```bash
conda create -n vm_optimizer python=3.9
conda activate vm_optimizer
pip install -r requirements.txt
streamlit run streamlit_app.py
```
</details>

<details>
<summary><b>Using Docker</b></summary>

```bash
# Build image
docker build -t vm-optimizer .

# Run container
docker run -p 8501:8501 vm-optimizer
```
</details>

### Verify Installation

```bash
python -c "import streamlit, sklearn, pandas, numpy, plotly; print('✅ All packages installed successfully!')"
```

---

## 📖 Usage Guide

### Method 1: Using Pre-Trained Models (Quick Start)

**Perfect for**: Testing and immediate use without training

1. **Launch Application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Navigate to VM Placement Optimizer** (sidebar menu)

3. **Configure System Parameters**
   - Set distance: 100 km
   - Set bandwidth: 500 MBps
   - Configure server load: 50%
   - Set cache hit ratio: 70%
   - Define CPU/RAM utilization
   - Select storage tier

4. **Define VMs and Servers**
   - Number of VMs: 4
   - VM CPU demands: [4, 6, 8, 4]
   - Number of Servers: 2
   - Server capacities: [16, 20]
   - Server costs: [$1.0, $1.2] per hour

5. **Run Optimization** and view results

**Example Configuration**:
```
📊 Small Office Setup
VMs: 3 (demands: 4, 2, 6 CPU)
Servers: 2 (capacities: 16, 32 CPU)
Costs: $1.00, $1.50 per hour
Distance: 50km | Bandwidth: 200MBps | Storage: SSD
```

### Method 2: Training Custom Models

**Perfect for**: Custom datasets and specific use cases

1. **Prepare Dataset** (CSV format):
   ```csv
   bandwidth_MBps,cpu_utilization_%,ram_utilization_%,request_size_MB,distance_km,server_load,cache_hit_ratio,storage_tier,latency_ms,egress_cost_per_gb
   100,45,60,500,150,0.3,0.7,SSD,25.4,0.025
   200,30,40,300,80,0.2,0.8,NVMe,15.2,0.015
   ```

2. **Upload and Train**
   - Go to "📊 Performance Prediction" tab
   - Upload your CSV file
   - Review dataset preview
   - Click "🚀 Train Models"
   - Compare model performance

3. **Use Trained Models**
   - Models are automatically saved
   - Switch to "VM Placement Optimizer" to use them

### Method 3: Command Line Training

**Perfect for**: Batch processing and automation

```bash
# Ensure dataset is at: data/vm_placement_cdn_dataset.csv
python train_model.py

# Models will be saved to: models/
```

### Dataset Requirements

**Required Columns**:
- `bandwidth_MBps` - Network bandwidth
- `cpu_utilization_%` - CPU usage percentage
- `ram_utilization_%` - Memory usage percentage
- `request_size_MB` - Request size
- `distance_km` - Geographic distance
- `server_load` - Server load ratio (0-1)
- `cache_hit_ratio` - Cache hit ratio (0-1)
- `storage_tier` - Storage type (HDD/SSD/NVMe)
- `latency_ms` - **Target**: Network latency
- `egress_cost_per_gb` - **Target**: Data egress cost

---

## 📊 Model Performance

### Latency Prediction Models

| Model | MAE (ms) | Status | Use Case |
|-------|----------|--------|----------|
| **LightGBM** | **5.37** | ✅ Best | Production - High Accuracy |
| XGBoost | 5.09 | ⭐ Excellent | Production - Balanced |
| Random Forest | 6.78 | ✓ Very Good | Training - Robust |
| Polynomial Regression | 57.47 | - Baseline | Comparison Only |

### Cost Prediction Models

| Model | MAE ($/GB) | Status | Use Case |
|-------|------------|--------|----------|
| **Linear Regression** | **0.001012** | ✅ Best | Production - Fast |
| **Ridge** | **0.001012** | ✅ Best | Production - Regularized |
| Polynomial Regression | 0.001013 | ⭐ Excellent | Training - Complex |
| LightGBM | 0.001017 | ⭐ Excellent | Production - Accurate |

**Performance Metrics**:
- Lower MAE = Better prediction accuracy
- ✅ Best: Optimal for production deployment
- ⭐ Excellent: Suitable for production with trade-offs
- ✓ Very Good: Good for training and development

---

## 📁 Project Structure

```
vm-placement-cdn-optimizer/
├── 📄 streamlit_app.py          # Main web application interface
├── 📄 cdn_optimizer.py          # Core optimization algorithms
├── 📄 train_model.py            # ML model training pipeline
├── 📄 utils.py                  # Utility functions
├── 📄 sample_dataset.py         # Sample data generator
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
├── 📄 LICENSE                   # MIT License
│
├── 📁 models/                   # Pre-trained ML models
│   ├── best_latency.joblib      # Best latency model (LightGBM)
│   ├── best_cost.joblib         # Best cost model (Linear)
│   ├── latency_results.csv      # Training results (latency)
│   └── cost_results.csv         # Training results (cost)
│
├── 📁 data/                     # Dataset directory
│   └── vm_placement_cdn_dataset.csv
│
└── 📁 __pycache__/              # Python cache (auto-generated)
```

### Key Components

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main UI with 3 tabs: Home, Prediction, Optimizer |
| `cdn_optimizer.py` | ILP/Greedy optimization + ML integration |
| `train_model.py` | Model training, evaluation, and selection |
| `utils.py` | Data preprocessing and feature engineering |

---

## ⚙️ Configuration

### System Parameters

Configure these in the Streamlit UI:

```python
# Performance Parameters
distance_km = 100           # Geographic distance
bandwidth_MBps = 500        # Network bandwidth
server_load = 0.5           # Current server load (0-1)
cache_hit_ratio = 0.7       # Cache efficiency (0-1)
cpu_utilization = 0.5       # CPU usage (0-1)
ram_utilization = 0.6       # RAM usage (0-1)
request_size_MB = 500       # Average request size
storage_tier = 'SSD'        # HDD/SSD/NVMe
```

### VM and Server Configuration

```python
# VM Configuration
num_vms = 4
vm_demands = [4, 6, 8, 4]   # CPU requirements per VM

# Server Configuration
num_servers = 2
server_capacities = [16, 20] # CPU capacity per server
server_costs = [1.0, 1.2]    # $/hour per server
```

### Model Training Configuration

Edit `train_model.py` for custom settings:

```python
# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters (example for XGBoost)
xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'random_state': 42
}
```

---

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended)

**Steps**:
1. Fork this repository to your GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Click "Deploy"

**Benefits**: Free hosting, auto-deployment on push, HTTPS included

### Option 2: Docker Deployment

```bash
# Build Docker image
docker build -t vm-optimizer:latest .

# Run container
docker run -d \
  -p 8501:8501 \
  --name vm-optimizer \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  vm-optimizer:latest

# Check logs
docker logs -f vm-optimizer
```

### Option 3: Cloud Platforms

<details>
<summary><b>AWS EC2</b></summary>

```bash
# On EC2 instance
sudo yum update -y
sudo yum install python3 python3-pip git -y

git clone https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization.git
cd VM_Placement_and_CDN_Optimization

pip3 install -r requirements.txt

# Run with nohup
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
```
</details>

<details>
<summary><b>Google Cloud Run</b></summary>

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/vm-optimizer
gcloud run deploy vm-optimizer \
  --image gcr.io/PROJECT_ID/vm-optimizer \
  --platform managed \
  --port 8501
```
</details>

<details>
<summary><b>Heroku</b></summary>

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

Create `Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

Deploy:
```bash
heroku create your-app-name
git push heroku main
```
</details>

---

## 🐛 Troubleshooting

### Installation Issues

<details>
<summary><b>Package installation fails</b></summary>

```bash
# Solution 1: Upgrade pip
python -m pip install --upgrade pip

# Solution 2: Install with no cache
pip install -r requirements.txt --no-cache-dir

# Solution 3: Install individually
pip install streamlit scikit-learn pandas numpy plotly
pip install xgboost lightgbm joblib pulp
```
</details>

<details>
<summary><b>LightGBM fails on macOS</b></summary>

```bash
# For Intel Macs
brew install cmake
pip install lightgbm

# For Apple Silicon (M1/M2/M3)
brew install cmake libomp
pip install --no-use-pep517 lightgbm
```
</details>

<details>
<summary><b>XGBoost installation error</b></summary>

```bash
# Try conda installation
conda install -c conda-forge xgboost

# Or build from source
pip install xgboost --no-binary xgboost
```
</details>

### Runtime Issues

<details>
<summary><b>"Models not found" error</b></summary>

```bash
# Check models directory
ls -la models/

# Create if missing
mkdir models

# Train models
python train_model.py
```
</details>

<details>
<summary><b>Port already in use</b></summary>

```bash
# Find process using port 8501
lsof -i :8501

# Kill process (replace PID)
kill -9 <PID>

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```
</details>

<details>
<summary><b>Optimization fails</b></summary>

**Common causes**:
- ❌ Total VM demands exceed total server capacity
- ❌ Negative or zero VM demands
- ❌ Invalid server capacities

**Solutions**:
```python
# Ensure capacity > demand
total_demand = sum(vm_demands)      # e.g., 18 CPU
total_capacity = sum(capacities)    # e.g., 36 CPU

# Add buffer
assert total_capacity >= total_demand * 1.2
```
</details>

<details>
<summary><b>Dataset upload fails</b></summary>

**Checklist**:
- ✅ File format is CSV
- ✅ File size < 200MB
- ✅ All required columns present
- ✅ No special characters in headers
- ✅ No empty rows at start

**Validate**:
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.head())
print(df.columns.tolist())
```
</details>

### Performance Issues

<details>
<summary><b>Slow model training</b></summary>

Edit `train_model.py`:
```python
# Reduce estimators
"RandomForest": RandomForestRegressor(
    n_estimators=50,  # Instead of 100
    n_jobs=-1
),
"XGBoost": XGBRegressor(
    n_estimators=50,
    n_jobs=-1
)
```
</details>

<details>
<summary><b>Memory issues with large datasets</b></summary>

```python
# Reduce data size
df_sample = df.sample(frac=0.5, random_state=42)

# Or use chunking
for chunk in pd.read_csv('file.csv', chunksize=1000):
    process_chunk(chunk)
```
</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/VM_Placement_and_CDN_Optimization.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make changes and test
streamlit run streamlit_app.py
```

### Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Update README for new features
- Test thoroughly before submitting

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary**: You can use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided you include the copyright notice and license.

---

## 🙏 Acknowledgments

- **Streamlit** - Web application framework
- **PuLP** - Linear programming optimization
- **scikit-learn, XGBoost, LightGBM** - Machine learning frameworks
- **Plotly** - Interactive visualizations
- **Open Source Community** - For inspiration and support

---

## 📞 Contact & Support

**Developer**: Vinayak Joshi  
**GitHub**: [@vinayakjoshi04](https://github.com/vinayakjoshi04)  
**Project**: [VM Placement & CDN Optimizer](https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization)

### Get Help

- 🐛 **Bug Reports**: [Open an issue](https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization/issues)
- 💡 **Feature Requests**: [Submit a request](https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization/issues)
- 📖 **Documentation**: Check this README and code comments
- 💬 **Discussions**: Use GitHub Discussions for questions

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐️!

**Star** the repository to help others discover it.

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/vinayakjoshi04/VM_Placement_and_CDN_Optimization?style=social)
![GitHub forks](https://img.shields.io/github/forks/vinayakjoshi04/VM_Placement_and_CDN_Optimization?style=social)
![GitHub issues](https://img.shields.io/github/issues/vinayakjoshi04/VM_Placement_and_CDN_Optimization)
![GitHub license](https://img.shields.io/github/license/vinayakjoshi04/VM_Placement_and_CDN_Optimization)

---

<div align="center">

**[🔝 Back to Top](#-vm-placement--cdn-optimizer)**

Made with ❤️ by [Vinayak Joshi](https://github.com/vinayakjoshi04)

*Optimizing infrastructure, one VM at a time* 🚀

</div>