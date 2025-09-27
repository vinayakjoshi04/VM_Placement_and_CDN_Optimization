import pandas as pd
import numpy as np
import random
import os

def generate_sample_dataset(n_rows=50, filename="sample_dataset.csv"):
    storage_tiers = ["HDD", "SSD", "NVMe"]

    # ensure data folder exists
    os.makedirs("data", exist_ok=True)

    data = []
    for _ in range(n_rows):
        distance = random.randint(100, 10000)                     # km
        bandwidth = random.randint(50, 1000)                      # MBps
        server_load = round(random.uniform(0.1, 0.95), 2)         # fraction
        cache_hit = round(random.uniform(0.4, 0.95), 2)           # fraction
        cpu_util = round(random.uniform(0.2, 0.95), 2)            # fraction
        ram_util = round(random.uniform(0.3, 0.95), 2)            # fraction
        request_size = random.randint(100, 5000)                  # MB
        storage_tier = random.choice(storage_tiers)

        # Synthetic performance metrics
        latency = (
            distance / bandwidth * 10
            + (1 - cache_hit) * 50
            + cpu_util * 20
            + ram_util * 15
            + random.gauss(0, 5)  # noise
        )
        latency = max(10, round(latency, 2))

        cost = (
            0.02
            + 0.01 * (1 - cache_hit)
            + 0.005 * (server_load)
            + (0.01 if storage_tier == "HDD" else 0.005 if storage_tier == "SSD" else 0.003)
            + random.uniform(-0.002, 0.002)  # noise
        )
        cost = round(max(0.01, cost), 4)

        data.append([
            distance, bandwidth, server_load, cache_hit, cpu_util, ram_util,
            request_size, storage_tier, latency, cost
        ])

    df = pd.DataFrame(data, columns=[
        "distance_km", "bandwidth_MBps", "server_load", "cache_hit_ratio",
        "cpu_utilization", "ram_utilization", "request_size_MB", "storage_tier",
        "latency_ms", "egress_cost_per_gb"
    ])

    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Dataset generated and saved as {filepath}")

# Example usage
if __name__ == "__main__":
    generate_sample_dataset(n_rows=20000, filename="vm_placement_cdn_dataset.csv")  # create 100 rows
