import pulp
import pandas as pd
import os
import joblib
from sklearn.pipeline import Pipeline


latency_model = None
cost_model = None
MODEL_DIR = "models"


def load_models():
    """Load best latency and cost models from training phase."""
    models = {}
    latency_path = os.path.join(MODEL_DIR, "best_latency.joblib")
    cost_path = os.path.join(MODEL_DIR, "best_cost.joblib")

    if os.path.exists(latency_path):
        try:
            models["latency"] = joblib.load(latency_path)
        except Exception as e:
            print(f"Warning: Could not load latency model: {e}")
    
    if os.path.exists(cost_path):
        try:
            models["cost"] = joblib.load(cost_path)
        except Exception as e:
            print(f"Warning: Could not load cost model: {e}")
    
    return models


def set_models(models):
    """Inject ML models (from training or session_state)."""
    global latency_model, cost_model
    print(f"=== DEBUG: set_models called ===")
    print(f"Input models dict: {models}")
    print(f"Models keys: {list(models.keys()) if models else 'None'}")
    
    latency_model = models.get("latency")

    cost_model = models.get("cost") or models.get("egress")
    
    print(f"Set latency_model: {latency_model}")
    print(f"Set cost_model: {cost_model}")
    print(f"Latency model type: {type(latency_model)}")
    print(f"Cost model type: {type(cost_model)}")



try:
    _loaded = load_models()
    set_models(_loaded)
except Exception as e:
    print(f"Warning: Could not auto-load models on import: {e}")


def get_model_name(model):
    """Return the model or pipeline's final estimator name."""
    try:
        if isinstance(model, Pipeline):
            final_est = model.steps[-1][1]
            return f"Pipeline({final_est.__class__.__name__})"
        return model.__class__.__name__
    except Exception:
        return "UnknownModel"


def predict_latency_cost(features: dict):
    """Predict latency & cost using ML models + return model names."""
    print(f"=== DEBUG: predict_latency_cost called ===")
    print(f"Latency model exists: {latency_model is not None}")
    print(f"Cost model exists: {cost_model is not None}")
    
    if not latency_model or not cost_model:
        print("Warning: Models not loaded. Using fallback values.")
        print(f"Latency model: {latency_model}")
        print(f"Cost model: {cost_model}")
        return 50.0, 0.05, "NoModel", "NoModel"
    
    try:
        features_copy = features.copy()
        
        for key, value in features_copy.items():
            if key != 'storage_tier':
                features_copy[key] = float(value)
        
        print(f"Features for prediction: {features_copy}")
        X = pd.DataFrame([features_copy])
        print(f"DataFrame shape: {X.shape}")
        print(f"DataFrame columns: {list(X.columns)}")
        
        latency_pred = None
        latency_name = "ErrorModel"
        try:
            latency_pred = latency_model.predict(X)[0]
            latency_name = get_model_name(latency_model)
        except Exception as e:
            print(f"Latency prediction error: {e}")
            latency_pred = 50.0
        
        cost_pred = None
        cost_name = "ErrorModel"
        try:
            cost_pred = cost_model.predict(X)[0]
            cost_name = get_model_name(cost_model)
        except Exception as e:
            print(f"Cost prediction error: {e}")
            cost_pred = 0.05

        return float(latency_pred), float(cost_pred), latency_name, cost_name
        
    except Exception as e:
        print(f"[Predict Error] {e}")
        return 50.0, 0.05, "ErrorModel", "ErrorModel"


def ilp_vm_optimizer(vm_names, server_names, demands, capacities, server_cost):
    """Solve VM placement using Integer Linear Programming (ILP)."""
    try:
        if not vm_names or not server_names:
            return None
        
        if any(demands[vm] <= 0 for vm in vm_names):
            print("Warning: Invalid VM demands (must be > 0)")
            return None
            
        if any(capacities[srv] <= 0 for srv in server_names):
            print("Warning: Invalid server capacities (must be > 0)")
            return None
        
        prob = pulp.LpProblem("VM_Placement", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", (vm_names, server_names), cat="Binary")

        prob += pulp.lpSum(x[vm][srv] * server_cost[srv] for vm in vm_names for srv in server_names)

        for vm in vm_names:
            prob += pulp.lpSum(x[vm][srv] for srv in server_names) == 1

        for srv in server_names:
            prob += pulp.lpSum(demands[vm] * x[vm][srv] for vm in vm_names) <= capacities[srv]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"ILP solver status: {pulp.LpStatus[prob.status]}")
            return None

        placement = {}
        for vm in vm_names:
            for srv in server_names:
                if pulp.value(x[vm][srv]) == 1:
                    placement[vm] = srv
                    break
        
        if len(placement) != len(vm_names):
            print("Warning: Not all VMs were placed by ILP solver")
            return None
            
        return placement
        
    except Exception as e:
        print(f"ILP solver error: {e}")
        return None


def greedy_vm_optimizer(vm_names, server_names, demands, capacities, server_cost):
    """Greedy fallback: sort VMs by demand and place on cheapest feasible server."""
    try:
        if not vm_names or not server_names:
            return None
            
        placement = {}
        available = capacities.copy()
        
        sorted_vms = sorted(vm_names, key=lambda v: demands[v], reverse=True)
        
        for vm in sorted_vms:
            if demands[vm] <= 0:
                continue
                
            feasible = [s for s in server_names if available[s] >= demands[vm]]
            
            if not feasible:
                print(f"No feasible server found for VM {vm} (demand: {demands[vm]})")
                return None
            
            best_server = min(feasible, key=lambda s: (server_cost[s], -available[s]))
            placement[vm] = best_server
            available[best_server] -= demands[vm]
        
        return placement
        
    except Exception as e:
        print(f"Greedy optimizer error: {e}")
        return None


def hybrid_vm_optimizer(vm_names, server_names, demands, capacities, server_cost):
    """Try ILP first, fallback to Greedy."""

    if not vm_names or not server_names:
        print("Error: Empty VM or server lists")
        return None
    
    if any(vm not in demands for vm in vm_names):
        print("Error: Missing demand values for some VMs")
        return None
        
    if any(srv not in capacities for srv in server_names):
        print("Error: Missing capacity values for some servers")
        return None
        
    if any(srv not in server_cost for srv in server_names):
        print("Error: Missing cost values for some servers")
        return None
    
    placement = ilp_vm_optimizer(vm_names, server_names, demands, capacities, server_cost)
    
    if placement is None:
        print("ILP failed, trying greedy approach...")
        placement = greedy_vm_optimizer(vm_names, server_names, demands, capacities, server_cost)
    
    return placement


def vm_placement_with_performance(vm_names, server_names, demands, capacities,
                                  server_cost, perf_params):
    """
    Run VM placement + ML latency/cost prediction.
    Distribute latency/cost across servers by utilization share.
    """
    try:
        placement = hybrid_vm_optimizer(vm_names, server_names, demands, capacities, server_cost)
        if placement is None:
            print("Error: VM placement optimization failed")
            return None

        latency_pred, cost_pred, latency_name, cost_name = predict_latency_cost(perf_params)

        util = {srv: 0 for srv in server_names}
        for vm, srv in placement.items():
            util[srv] += demands[vm]

        total_used = sum(util.values())
        if total_used == 0:
            print("Warning: No resources used")
            total_used = 1

        server_vm_map = {srv: [] for srv in server_names}
        for vm, srv in placement.items():
            server_vm_map[srv].append(vm)

        server_metrics = {}
        for srv in server_names:
            utilization_share = (util[srv] / total_used) if total_used > 0 else 0
            utilization_percent = (100 * util[srv] / capacities[srv]) if capacities[srv] > 0 else 0
            
            server_metrics[srv] = {
                "latency": round(latency_pred * utilization_share, 4) if latency_pred else 0,
                "cost": round(cost_pred * utilization_share, 4) if cost_pred else 0,
                "utilization": round(utilization_percent, 2),
                "vms": server_vm_map[srv]
            }

        return {
            "placement": placement,
            "placement_server_map": server_vm_map,
            "latency": round(latency_pred, 4) if latency_pred else 0,
            "cost": round(cost_pred, 4) if cost_pred else 0,
            "latency_model": latency_name,
            "cost_model": cost_name,
            "server_metrics": server_metrics
        }
        
    except Exception as e:
        print(f"Error in vm_placement_with_performance: {e}")
        return None