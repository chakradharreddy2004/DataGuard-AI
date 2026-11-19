"""
Synthetic Dirty Dataset Generator for DataGuard AI Demo & Testing.
Seeds realistic anomalies:
- Missing values (nulls, empty strings, whitespace-only)
- Exact full-row duplicates
- Conflicting business-key duplicates
- Casing inconsistencies & whitespace padding
- City aliases and fuzzy spelling variations
- Numerical statistical outliers (IQR / Z-score)
"""
import random
import pandas as pd
import numpy as np

def generate_dirty_sample_dataframe(num_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    cities_pool = [
        "New York", "NYC", "New York City",
        "San Francisco", "SF", "San Fran",
        "Los Angeles", "LA", "Los Angeles CA",
        "Chicago", "Chicagoo",
        "Houston", "Austin", "Boston"
    ]
    city_weights = [0.15, 0.05, 0.03, 0.12, 0.04, 0.02, 0.14, 0.05, 0.02, 0.12, 0.02, 0.08, 0.08, 0.08]

    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

    records = []
    for i in range(1, num_rows + 1):
        cust_id = f"CUST-{1000 + i}"
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        
        # Inject casing and whitespace variations
        rand_style = random.random()
        if rand_style < 0.08:
            full_name = f"{fn.lower()} {ln.lower()}"
        elif rand_style < 0.15:
            full_name = f"{fn.upper()} {ln.upper()}"
        elif rand_style < 0.22:
            full_name = f"  {fn}  {ln}   "
        else:
            full_name = f"{fn} {ln}"

        email = f"{fn.lower()}.{ln.lower()}{i % 50}@example.com"
        city = random.choices(cities_pool, weights=city_weights)[0]
        
        # Missing values in city or age
        if random.random() < 0.06:
            city = ""
        elif random.random() < 0.04:
            city = "   "

        age = int(np.random.normal(38, 12))
        age = max(18, min(80, age))
        if random.random() < 0.05:
            age = None

        income = round(float(np.random.normal(65000, 18000)), 2)
        income = max(20000.0, income)

        purchase_amount = round(float(np.random.exponential(scale=120) + 15), 2)
        
        # Inject extreme statistical outliers
        if random.random() < 0.015:
            purchase_amount = round(random.choice([15000.0, 24500.0, 48000.0, 99999.0]), 2)

        records.append({
            "customer_id": cust_id,
            "full_name": full_name,
            "email": email,
            "city": city,
            "age": age,
            "annual_income": income,
            "purchase_amount": purchase_amount,
            "membership_status": random.choice(["Gold", "Silver", "Bronze", "Standard"])
        })

    df = pd.DataFrame(records)

    # 1. Inject exact full-row duplicates (5 rows copied)
    exact_dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, exact_dup_rows], ignore_index=True)

    # 2. Inject conflicting business-key duplicates (same customer_id, different email/city)
    key_dup_row_1 = df.iloc[10].copy()
    key_dup_row_1["email"] = "conflicted.email@different.org"
    key_dup_row_1["city"] = "Boston"

    key_dup_row_2 = df.iloc[25].copy()
    key_dup_row_2["email"] = "altered.id@spoof.net"

    df = pd.concat([df, pd.DataFrame([key_dup_row_1, key_dup_row_2])], ignore_index=True)

    return df

def generate_dirty_sample_csv() -> bytes:
    df = generate_dirty_sample_dataframe()
    return df.to_csv(index=False).encode("utf-8")
