# VM Placement & CDN Optimizer

A machine learning-powered system for optimizing virtual machine placement and CDN performance prediction. This application combines advanced optimization algorithms with predictive models to minimize latency and costs in distributed computing environments.

## Table of Contents
- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Getting Started](#getting-started)
- [Complete Setup Instructions](#complete-setup-instructions)
- [How to Use the Project](#how-to-use-the-project)
- [Features](#features)
- [Data Format](#data-format)
- [Model Performance](#model-performance)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

The VM Placement & CDN Optimizer provides:
- **Performance Prediction**: ML models trained on real-world data to predict latency and egress costs
- **VM Placement Optimization**: Intelligent placement of virtual machines across servers using mixed-integer programming
- **Real-time Analytics**: Interactive dashboard for monitoring system performance and resource utilization

## Project Workflow

### 1. Data Flow Architecture
```
Raw Dataset (CSV) → Data Preprocessing → Model Training → Model Evaluation → Best Model Selection → Deployment
                                           ↓
VM Configuration → Performance Parameters → ML Prediction → Optimization Algorithm → Optimal Placement
```

### 2. Complete Workflow Steps

**Phase 1: Model Training & Preparation**
1. **Dataset Preparation**: Load and validate the `vm_placement_cdn_dataset.csv`
2. **Feature Engineering**: Process numerical and categorical features
3. **Model Training**: Train 8 different ML algorithms (Linear Regression, Ridge, Lasso, Polynomial, Random Forest, SVR, XGBoost, LightGBM)
4. **Model Evaluation**: Compare performance using Mean Absolute Error (MAE)
5. **Model Selection**: Automatically select best performing models
6. **Model Persistence**: Save best models as `.joblib` files

**Phase 2: Real-time Optimization**
1. **Input Collection**: Gather VM requirements and server specifications
2. **Performance Prediction**: Use trained ML models to predict latency and costs
3. **Constraint Setup**: Define capacity, demand, and cost constraints
4. **Optimization**: Run hybrid ILP + greedy algorithm
5. **Result Generation**: Provide optimal VM-to-server assignments
6. **Visualization**: Display results through interactive dashboard

**Phase 3: Analysis & Monitoring**
1. **Performance Tracking**: Monitor system metrics and resource utilization
2. **Model Comparison**: Analyze different algorithm performance
3. **Historical Analysis**: Track optimization results over time

## Features

### 🔮 ML-Powered Performance Prediction
- Train multiple ML models (Linear Regression, Random Forest, XGBoost, LightGBM, SVR) on your dataset
- Compare model performance with automated evaluation metrics
- Real-time prediction of latency (ms) and egress costs ($/GB)

### ⚡ VM Placement Optimization
- **Hybrid Optimization**: Combines Integer Linear Programming (ILP) with greedy fallback algorithms
- **Multi-objective**: Optimizes for both performance and cost simultaneously
- **Constraint Handling**: Respects server capacity and VM resource requirements
- **Real-time Results**: Instant optimization with detailed placement recommendations

### 📊 Interactive Analytics
- Visual performance comparisons across different ML models
- Server utilization monitoring and resource allocation insights
- Historical performance tracking and trend analysis

## 🚀 Live Demo

**Try the application now:** [VM Placement & CDN Optimizer](https://vmplacementandcdnoptimization-mnzkzoya3ysevnq3h3qsj6.streamlit.app/)

Experience the full functionality without any installation - the live demo includes pre-trained models and all optimization features.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: At least 1GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

## Complete Setup Instructions

### Step 1: Clone the Repository

**Option A: Using Git (Recommended)**
```bash
# Clone the repository
https://github.com/vinayakjoshi04/VM_Placement_and_CDN_Optimization.git

# Navigate to the project directory
cd VM_Placement_and_CDN_Optimization
```

**Option B: Download ZIP**
1. Go to the GitHub repository page
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your desired location
5. Open terminal/command prompt and navigate to the extracted folder

### Step 2: Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create a virtual environment
python -m venv vm_optimizer_env

# Activate the virtual environment
# On Windows:
vm_optimizer_env\Scripts\activate
# On macOS/Linux:
source vm_optimizer_env/bin/activate
```

**Option B: Using Conda**
```bash
# Create conda environment
conda create -n vm_optimizer python=3.9

# Activate the environment
conda activate vm_optimizer
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

**If you encounter installation issues, try:**
```bash
# Update pip first
pip install --upgrade pip

# Install with no cache
pip install -r requirements.txt --no-cache-dir

# For Apple M1/M2 users, if you get errors:
pip install --no-use-pep517 lightgbm
```

### Step 4: Verify Setup

```bash
# Test if all packages are working
python -c "import streamlit, sklearn, pandas, numpy, plotly; print('All packages installed successfully!')"
```

### Step 5: Run the Application

```bash
# Start the Streamlit application
streamlit run streamlit_app.py

# The app will automatically open in your browser at http://localhost:8501
```

**Alternative: Specify port manually**
```bash
streamlit run streamlit_app.py --server.port 8502
```

## How to Use the Project

### Method 1: Using Pre-trained Models (Quickest)

**Step 1: Start the Application**
```bash
streamlit run streamlit_app.py
```

**Step 2: Navigate to VM Placement Optimizer**
- Click on "VM Placement Optimizer" in the sidebar
- The app will automatically load pre-trained models

**Step 3: Configure Your System**
1. **Set Performance Parameters:**
   - Distance: 100km (example)
   - Bandwidth: 100 MBps
   - Server Load: 50%
   - Cache Hit Ratio: 70%
   - CPU Utilization: 50%
   - RAM Utilization: 60%
   - Request Size: 500MB
   - Storage Tier: SSD

2. **Configure VMs and Servers:**
   - Number of VMs: 6
   - Number of Servers: 3
   - Set VM demands (CPU requirements)
   - Set server capacities and costs

**Step 4: Run Optimization**
- Click "Run Optimization"
- View results: optimal placement, predicted latency/cost, server utilization

### Method 2: Training Your Own Models

**Step 1: Prepare Your Dataset**
Create a CSV file with the required columns:
```csv
bandwidth_MBps,cpu_utilization_%,ram_utilization_%,request_size_MB,distance_km,server_load,cache_hit_ratio,storage_tier,latency_ms,egress_cost_per_gb
100,45,60,500,150,0.3,0.7,SSD,25.4,0.025
200,30,40,300,80,0.2,0.8,NVMe,15.2,0.015
```

**Step 2: Upload and Train**
1. Go to "Performance Prediction" tab
2. Upload your CSV dataset
3. Review the dataset preview
4. Click "Train Models"
5. Compare model performance results

**Step 3: Use Trained Models**
- Navigate to "VM Placement Optimizer"
- Your newly trained models will be automatically used

### Method 3: Standalone Model Training

**For Advanced Users - Command Line Training:**
```bash
# Ensure your dataset is in the correct location
# Place your CSV file at: data/vm_placement_cdn_dataset.csv

# Run the training script
python train_model.py

# This will create trained models in the models/ directory
```

### Complete Usage Examples

#### Example 1: Small Office Setup
```
VMs: 3 (VM1: 4 CPU, VM2: 2 CPU, VM3: 6 CPU)
Servers: 2 (Server1: 16 CPU, $1.00/hour; Server2: 32 CPU, $1.50/hour)
Parameters: Distance=50km, Bandwidth=200MBps, SSD storage
```

#### Example 2: Data Center Configuration
```
VMs: 10 (varying CPU demands: 4-16 CPU each)
Servers: 5 (varying capacities: 64-128 CPU each)
Parameters: Distance=10km, Bandwidth=1000MBps, NVMe storage
```

### Step-by-Step Workflow Example

1. **Launch Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **System Status Check:**
   - Look at sidebar "System Status"
   - Verify models are "Online" (green status)

3. **Input Configuration:**
   - Set distance: 100km
   - Set bandwidth: 500 MBps
   - Configure 4 VMs with demands: [4, 6, 8, 4] CPU
   - Configure 2 servers with capacities: [16, 20] CPU
   - Set server costs: [$1.0, $1.2] per hour

4. **Run Optimization:**
   - Click "Run Optimization"
   - Wait for results (typically 1-5 seconds)

5. **Analyze Results:**
   - Review predicted latency and cost
   - Check VM-to-server assignments
   - Monitor server utilization percentages
   - Examine performance visualizations

6. **Iterate and Optimize:**
   - Adjust parameters to see impact
   - Try different VM/server configurations
   - Compare multiple optimization scenarios

### Using Pre-trained Models

The application comes with pre-trained models that were trained on a comprehensive dataset:
- `models/best_latency.joblib` - Optimized for latency prediction (MAE: 5.37ms)
- `models/best_cost.joblib` - Optimized for cost prediction (MAE: $0.001/GB)

These models are automatically loaded when the application starts.

## Project File Structure

```
vm-placement-cdn-optimizer/
├── streamlit_app.py          # Main web application
├── cdn_optimizer.py          # Optimization algorithms & ML integration
├── train_model.py           # Model training pipeline
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── models/                 # Pre-trained ML models
│   ├── best_latency.joblib    # Best latency prediction model
│   ├── best_cost.joblib       # Best cost prediction model
│   ├── latency_results.csv    # Training results for latency models
│   └── cost_results.csv       # Training results for cost models
├── data/                   # Dataset directory (optional)
│   └── vm_placement_cdn_dataset.csv
└── notebooks/              # Jupyter notebooks (optional)
    └── analysis.ipynb
```

## Detailed Feature Walkthrough

### 1. Performance Prediction Module

**Upload Your Dataset**
- Navigate to the "📊 Performance Prediction" tab
- Upload a CSV file with the required columns (see Data Format section)
- View dataset preview and column information
- Click "🚀 Train Models" to create custom models from your data

**Model Training Process**
1. **Data Validation**: System checks for required columns and data quality
2. **Feature Engineering**: Automatic preprocessing of numerical and categorical features
3. **Model Training**: Parallel training of 8 different algorithms
4. **Performance Evaluation**: Comparison using Mean Absolute Error (MAE)
5. **Model Selection**: Automatic selection of best performing models
6. **Results Display**: Interactive charts showing model performance

**What You'll See**
- Real-time training progress bar
- Performance comparison tables for latency and cost models
- Interactive bar charts showing MAE scores
- Best model identification with performance metrics

### 2. VM Placement Optimizer Module

**Step-by-Step Configuration**

1. **Performance Parameters Input**
   - 🌍 **Distance (km)**: Geographic distance between client and server
   - 📶 **Bandwidth (MBps)**: Available network bandwidth
   - 🖥️ **Server Load (%)**: Current server utilization
   - 📂 **Cache Hit Ratio (%)**: Percentage of requests served from cache
   - ⚡ **CPU Utilization (%)**: Current CPU usage
   - 💾 **RAM Utilization (%)**: Current memory usage
   - 📦 **Request Size (MB)**: Average request size
   - 💽 **Storage Tier**: Storage type (HDD/SSD/NVMe)

2. **VM and Server Configuration**
   - **Number of VMs**: Specify how many virtual machines need placement (2-20)
   - **Number of Servers**: Define available servers (2-10)
   - **VM Demands**: Set CPU requirements for each VM
   - **Server Capacities**: Define maximum capacity for each server
   - **Server Costs**: Set hourly costs for each server

3. **Optimization Execution**
   - Click "🚀 Run Optimization" button
   - System runs hybrid ILP + greedy optimization algorithm
   - Results displayed in real-time

**Optimization Results Display**
- **Overall Metrics**: Total predicted latency and cost
- **VM Assignments**: Detailed table showing which VM goes to which server
- **Server Performance**: Utilization percentages, latency, and cost per server
- **Visualization**: Interactive bar chart of server utilization

### 3. Analytics Dashboard Module

**Performance Monitoring**
- Historical model performance comparisons
- Side-by-side analysis of latency vs cost models
- Interactive charts with configurable parameters
- Export functionality for further analysis

**Real-time Insights**
- System status indicators (model health, last update time)
- Resource utilization trends
- Optimization history tracking

## Data Format

Your CSV dataset should include the following columns:

### Required Target Columns
- `latency_ms` - Network latency in milliseconds
- `egress_cost_per_gb` - Data egress cost per GB in dollars

### Required Feature Columns
- `bandwidth_MBps` - Available bandwidth in MB/s
- `cpu_utilization_%` - CPU utilization percentage
- `ram_utilization_%` - RAM utilization percentage
- `request_size_MB` - Request size in megabytes
- `distance_km` - Geographic distance in kilometers
- `server_load` - Server load ratio (0.0 to 1.0)
- `cache_hit_ratio` - Cache hit ratio (0.0 to 1.0)
- `storage_tier` - Storage type (HDD, SSD, NVMe)

### Example Dataset Row
```csv
bandwidth_MBps,cpu_utilization_%,ram_utilization_%,request_size_MB,distance_km,server_load,cache_hit_ratio,storage_tier,latency_ms,egress_cost_per_gb
100,45,60,500,150,0.3,0.7,SSD,25.4,0.025
```

## Model Performance

### Current Best Models (Pre-trained)

**Latency Prediction Models:**
| Model | MAE (ms) | Performance |
|-------|----------|-------------|
| LightGBM | 5.37 | Best |
| XGBoost | 5.09 | Excellent |
| Random Forest | 6.78 | Very Good |
| Polynomial Regression | 57.47 | Baseline |

**Cost Prediction Models:**
| Model | MAE ($/GB) | Performance |
|-------|------------|-------------|
| Linear Regression | 0.001012 | Best |
| Ridge | 0.001012 | Best |
| Polynomial Regression | 0.001013 | Excellent |
| LightGBM | 0.001017 | Excellent |

## Architecture

### Core Components

1. **streamlit_app.py** - Main web application interface
2. **cdn_optimizer.py** - Optimization algorithms and ML model integration
3. **train_model.py** - Model training pipeline and evaluation
4. **utils.py** - Utility functions for data preprocessing
5. **models/** - Directory containing pre-trained ML models

### Optimization Algorithms

**Integer Linear Programming (ILP)**
- Primary optimization method using PuLP solver
- Guarantees optimal solutions for placement problems
- Handles complex constraints and multiple objectives

**Greedy Fallback Algorithm**
- Backup optimization when ILP fails or times out
- Fast approximate solutions for large-scale problems
- Sorts VMs by demand and places on cheapest feasible servers

### Machine Learning Pipeline

**Data Preprocessing**
- Automatic feature scaling using StandardScaler
- One-hot encoding for categorical variables
- Polynomial feature generation for complex relationships

**Model Training**
- Cross-validation with 80/20 train-test split
- Automated hyperparameter tuning
- Performance evaluation using Mean Absolute Error

**Model Selection**
- Automatic selection of best-performing models
- Support for multiple ML frameworks (scikit-learn, XGBoost, LightGBM)
- Fallback handling for missing dependencies

## Configuration

### Environment Variables
- `MODEL_DIR` - Directory for storing trained models (default: "models/")
- `DATA_PATH` - Path to training dataset (for standalone training)

### Model Parameters
Models can be fine-tuned by modifying parameters in `train_model.py`:
- Random Forest: `n_estimators`, `max_depth`
- XGBoost: `learning_rate`, `n_estimators`, `max_depth`
- LightGBM: `learning_rate`, `n_estimators`, `max_depth`

## Deployment

### Local Development
```bash
# Standard development mode
streamlit run streamlit_app.py

# Development with auto-reload
streamlit run streamlit_app.py --server.runOnSave true

# Custom port
streamlit run streamlit_app.py --server.port 8502
```

### Production Deployment Options

#### Option 1: Streamlit Cloud (Recommended for Demo)
1. Fork the repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

#### Option 2: Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

**Build and run Docker container:**
```bash
# Build the image
docker build -t vm-optimizer .

# Run the container
docker run -p 8501:8501 vm-optimizer
```

#### Option 3: AWS EC2 Deployment
```bash
# On your EC2 instance
sudo yum update -y
sudo yum install python3 python3-pip git -y

# Clone and setup
git clone https://github.com/yourusername/vm-placement-cdn-optimizer.git
cd vm-placement-cdn-optimizer
pip3 install -r requirements.txt

# Run with nohup for background execution
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
```

#### Option 4: Heroku Deployment
1. **Create Heroku-specific files:**

`Procfile`:
```
web: sh setup.sh && streamlit run streamlit_app.py
```

`setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

2. **Deploy:**
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit"

# Create Heroku app
heroku create your-vm-optimizer-app

# Deploy
git push heroku main
```

### Environment Configuration

**Development Environment Variables:**
```bash
# .env file
DEBUG=True
MODEL_DIR=models
DATA_PATH=data/vm_placement_cdn_dataset.csv
```

**Production Environment Variables:**
```bash
DEBUG=False
MODEL_DIR=/app/models
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
```

## Troubleshooting

### Installation Issues

**Problem: Package installation fails**
```bash
# Solution 1: Upgrade pip
python -m pip install --upgrade pip

# Solution 2: Install with no cache
pip install -r requirements.txt --no-cache-dir

# Solution 3: Install packages individually
pip install streamlit scikit-learn pandas numpy plotly joblib pulp
pip install xgboost lightgbm  # These might need special handling
```

**Problem: LightGBM installation fails on macOS**
```bash
# For Intel Macs
brew install cmake
pip install lightgbm

# For Apple Silicon (M1/M2)
brew install cmake libomp
pip install --no-use-pep517 lightgbm
```

**Problem: XGBoost installation fails**
```bash
# Alternative installation
conda install -c conda-forge xgboost
# OR
pip install xgboost --no-binary xgboost
```

### Runtime Issues

**Problem: "Models not found" error**
```bash
# Check if models directory exists
ls -la models/

# If missing, create directory
mkdir models

# Download pre-trained models or retrain
python train_model.py
```

**Problem: Port already in use**
```bash
# Find process using port 8501
lsof -i :8501

# Kill the process (replace PID)
kill -9 <PID>

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```

**Problem: Dataset upload fails**
- Check CSV format matches required schema
- Ensure file size is under Streamlit's 200MB limit
- Verify all required columns are present
- Check for special characters in column names

**Problem: Optimization fails**
- Verify server capacities exceed total VM demands
- Check that all VMs have positive resource requirements
- Ensure at least one feasible placement exists
- Try reducing the number of VMs/servers for testing

**Problem: Memory issues with large datasets**
```python
# Reduce model complexity in train_model.py
# For Random Forest:
RandomForestRegressor(n_estimators=50, max_depth=10)

# For XGBoost:
XGBRegressor(n_estimators=50, max_depth=4)
```

### Performance Optimization

**For Better Model Training Speed:**
```python
# In train_model.py, modify these parameters:
models = {
    "RandomForest": RandomForestRegressor(n_estimators=50, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=50, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(n_estimators=50, n_jobs=-1)
}
```

**For Faster Optimization:**
- Reduce number of VMs to under 10 for initial testing
- Use fewer servers (2-5) for quick results
- Ensure server capacities have some buffer above total demands

### Common Error Messages and Solutions

**Error: "ModuleNotFoundError: No module named 'lightgbm'"**
```bash
# Solution: Install missing package
pip install lightgbm
```

**Error: "Solver PuLP_CBC_CMD unavailable"**
```bash
# Solution: Install CBC solver
# Windows: Download from https://github.com/coin-or/Cbc
# macOS: brew install cbc
# Linux: sudo apt-get install coinor-cbc
```

**Error: "DataFrame constructor not properly called!"**
- Check that your CSV has proper headers
- Ensure no empty rows at the beginning
- Verify column names match exactly (case-sensitive)

## Advanced Configuration

### Custom Model Parameters

Edit `train_model.py` to customize model hyperparameters:

```python
# Example: Tuning XGBoost
"XGBoost": XGBRegressor(
    n_estimators=200,      # More trees
    learning_rate=0.05,    # Slower learning
    max_depth=8,           # Deeper trees
    subsample=0.8,         # Feature sampling
    random_state=42
)
```

### Adding New Models

Add custom models to the training pipeline:

```python
# In get_models() function in train_model.py
from sklearn.neural_network import MLPRegressor

models["NeuralNetwork"] = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    random_state=42
)
```

### Custom Optimization Constraints

Modify `cdn_optimizer.py` to add new constraints:

```python
# Example: Add maximum VMs per server constraint
max_vms_per_server = 3
for srv in server_names:
    prob += pulp.lpSum(x[vm][srv] for vm in vm_names) <= max_vms_per_server
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, bug reports, or feature requests, please open an issue in the GitHub repository.

## Acknowledgments

- Built with Streamlit for the web interface
- Optimization powered by PuLP and CBC solver
- Machine learning capabilities provided by scikit-learn, XGBoost, and LightGBM
- Interactive visualizations created with Plotly
