# Crop Production: Bounded Multi-dimensional Knapsack Problem

## Overview
This document outlines how a commercial farm models its seasonal planting decisions as a knapsack problem. The goal is to maximize total farm profit under strict resource constraints such as arable land, operating capital, and water allocation.

## Mathematical Formulation

### 1. Objective Function
Maximize the total profit generated from producing crop $j$, taking into account the fixed production costs $F_j$ if crop $j$ is selected ($y_j = 1$).

$$ \text{maximize} \; Z = \sum_{j=1}^n p_j x_j - \sum_{j=1}^n F_j y_j $$

### 2. Decision Variables
* $x_j \ge 0$: Continuous variable representing the exact number of hectares of crop $j$ to plant.
* $y_j \in \{0, 1\}$: Binary variable indicating if crop $j$ is selected for production (1) or skipped (0).

### 3. Constraints
* **Knapsack Capacity**: Total resource usage must not exceed the available limit $R_i$.
  $$ \sum_{j=1}^n a_{ij}x_j \leq R_i \quad \forall i $$
* **Minimum and Maximum Bounds**: If a crop is selected, its production must be within the bounds $[K_j, L_j]$. If it is not selected, production must be 0.
  $$ K_j y_j \leq x_j \leq L_j y_j \quad \forall j $$

---

## Python Implementation (using DOcplex)

### Step 1: Initialization & Data Loading
We start by initializing the model and loading the data via `pandas`.
```python
import pandas as st
from docplex.mp.model import Model

mdl = Model(name="crop_production")
df = st.read_csv("crop_production_data.csv", encoding="utf-8-sig")
df_res = st.read_csv("resource_limits_data.csv", encoding="utf-8-sig")
```

### Step 2: Extracting Parameters
Next, we extract the required parameters: profit per hectare ($p$), fixed fees ($F$), min bounds ($K$), max bounds ($L$), and resource limits ($R$).

```python
crops = df['crop_id'].tolist()
resources = df_res['resource_id'].tolist()

p = dict(zip(df["crop_id"], df["profit_p_j"]))
F = dict(zip(df["crop_id"], df["fixed_fee_F_j"]))
K = dict(zip(df["crop_id"], df["min_bound_K_j"]))
L = dict(zip(df["crop_id"], df["max_bound_L_j"]))
R = dict(zip(df_res["resource_id"], df_res["limit_R_i"]))

# Mapping the resource requirements a(i, j)
a = {}
for _, row in df.iterrows():
    j = int(row["crop_id"])
    a[(1, j)] = row["req_land"]     # Arable Land
    a[(2, j)] = row["req_capital"]  # Operating Capital
    a[(3, j)] = row["req_water"]    # Water Allocation
```

### Step 3: Defining Decision Variables
We declare our continuous $x$ variables and our binary $y$ selection variables.
```python
x = mdl.continuous_var_dict(crops, lb=0, name="hectares_planted")
y = mdl.binary_var_dict(crops, name="is_selected")
```

### Step 4: Objective and Constraints
We formulate the model with our objective equation and bound/capacity constraints.
```python
# Objective
total_profit = mdl.sum(p[j] * x[j] for j in crops)
total_fixed_cost = mdl.sum(F[j] * y[j] for j in crops)
mdl.maximize(total_profit - total_fixed_cost)

# Capacity Constraints (Sum of resources used <= Total Limit)
mdl.add_constraints(
    (mdl.sum(a[i, j] * x[j] for j in crops) <= R[i]) for i in resources
)

# Min/Max Bound Constraints (Forces x_j to 0 if y_j is 0)
for j in crops:
    mdl.add_constraint(x[j] >= K[j] * y[j], ctname=f"min_bound_crop_{j}")
    mdl.add_constraint(x[j] <= L[j] * y[j], ctname=f"max_bound_crop_{j}")
```

### Step 5: Solving and Displaying Results
Finally, we evaluate the optimal farming plan by calling `mdl.solve()` and iterating over the variables to present the chosen crops.
```python
print("Zach is solving.....")
solution = mdl.solve(log_output=True)

if solution:
    print(f"\nStatus: {mdl.get_solve_status()}")
    print(f"Total Profit (Z): ${solution.objective_value:,.2f}")
    
    crop_names = dict(zip(df["crop_id"], df["crop_name"]))
    
    selected_crops = []
    hectares = []
    revenues = []
    costs = []
    
    for j in crops:
        if solution.get_value(y[j]) > 0.5:
            qty = solution.get_value(x[j])
            selected_crops.append(crop_names[j])
            hectares.append(qty)
            revenues.append(qty * p[j])
            costs.append(F[j])
            
    results_df = st.DataFrame({
        'Selected Crop': selected_crops,
        'Hectares Planted': hectares,
        'Projected Revenue': revenues,
        'Production Cost': costs
    })
    
    display(results_df)
else:
    print("No feasible solution found.")
```
