import random
import scipy.stats as st
import pandas as pd
import numpy as np
import time
from cell_dumper import CellDumper
from itertools import combinations

# ==========================================
#  PRIVACY BUDGET 
# ==========================================
C = 5
EPSILON_TOTAL = 1.5 # 1- 10
EPSILON_QUERY = 0.5
EPSILON_REMAIN = EPSILON_TOTAL - EPSILON_QUERY  # = 1.0
#   Total cost of all six fields = 0.8+0.6+0.4+0.3+0.6+0.0 = 2.7 > 1.0,
#   so the Knapsack cannot afford all fields simultaneously — it must
#   make a genuine trade-off, which is exactly the behaviour the
#   optimisation layer is designed to demonstrate.
COSTS = {
    'tau_noise': 0.8,   
    'tau_sens':  0.6,   
    'tau_count': 0.4,  
    'tau_group_count': 0.3,
    'tau_row_contrib': 0.6,
    'tau_agg_comp':    0.0
}

# ==========================================
# DATA PROCESSING
# Returns:
# For each spatial cell in 10X10 grid: 
#   The # of taxis appeared & total how many times each taxi pinged
# ==========================================
def preprocess_data(filepath, grid_size = 10):
    """Loads the dataset and maps GPS coordinates to a 2D spatial grid."""
    print("Loading data...")
    df = pd.read_csv(filepath, sep=';', names=['taxi_id', 'timestamp', 'point'])
    df[['lat', 'lon']] = (
        df['point'].str.extract(r'POINT\(([\d\.]+) ([\d\.]+)\)').astype(float)
    )

    df['cell_x'] = pd.cut(df['lon'], bins=grid_size, labels=False)
    df['cell_y'] = pd.cut(df['lat'], bins=grid_size, labels=False)

    counts = df.groupby(['cell_x', 'cell_y', 'taxi_id']).size().reset_index(name='pings')
    # Calculate total pings per cell for regional classification
    cell_totals = counts.groupby(['cell_x', 'cell_y'])['pings'].sum().reset_index(name='total_pings')
    # Define thresholds (Top 20% = Urban, Middle 40-80% = Suburban, Bottom 40% = Rural)
    p80 = cell_totals['total_pings'].quantile(0.80)
    p40 = cell_totals['total_pings'].quantile(0.40)

    def classify_region(pings):
        if pings >= p80: return 'Urban'
        elif pings >= p40: return 'Suburban'
        else: return 'Rural'
        
    cell_totals['region'] = cell_totals['total_pings'].apply(classify_region)
    counts = counts.merge(cell_totals[['cell_x', 'cell_y', 'region']], on=['cell_x', 'cell_y'])
    return counts, cell_totals

# ==========================================
# BASE MECHANISM M(X)
# Returns: 
#   true query: f(X), 
#   Base DP: Y
#   Laplachian: true noise
# ==========================================
def run_base_mechanism(cell_data):
    """
    Y = f(X) + Lap(C / epsilon_query)
    Delta_f = C  (sensitivity of the clipped sum).
    """
    clipped_pings = cell_data['pings'].clip(upper=C) # clip each taxi's pings to max C = 5
    
    f_X = clipped_pings.sum()
    true_noise = np.random.laplace(0, C / EPSILON_QUERY)
    Y = f_X + true_noise
    return f_X, Y, true_noise


# ==========================================
# METADATA MECHANISMS K(X, Y)
# ==========================================
def generate_metadata(cell_data, f_X, Y, true_noise):
    """
    Generates all six candidate metadata fields.
    
    tau_agg_comp   : config string (zero cost)
    tau_noise      : K_noise(X,Y)  = (Y - f(X)) + Lap(C / Gamma_noise)
    tau_sens       : K_sens(X)     = (max-min of clipped pings) + Lap(C / Gamma_sens)
    tau_count      : K_count(X)    = Count(x_i > C) + Lap(1 / Gamma_count)
    tau_group_count: K_group(X)    = UniqueGroups(X) + Lap(1 / Gamma_group)
    tau_row_contrib: K_row(X)      = clip(max(x_i), 0, C) + Lap(C / Gamma_row)
    """
    traces = {}
    pings = cell_data['pings']
    clipped_pings = pings.clip(upper=C)
    n_taxis = len(cell_data) # Number of unique taxis in this cell

    traces['tau_noise'] = true_noise + np.random.laplace(0, C / COSTS['tau_noise'])

    # True sensitivity: computed on CLIPPED pings to match paper (§5.2, §6.2.1)
    sens_X = (clipped_pings.max() - clipped_pings.min()) if len(clipped_pings) > 1 else 0.0
    traces['tau_sens'] = sens_X + np.random.laplace(0, C / COSTS['tau_sens'])

    # Clipping count: strict > C 
    num_clipped = (pings > C).sum()
    traces['tau_count'] = num_clipped + np.random.laplace(0, 1.0 / COSTS['tau_count'])

    # Intermediate Group Count (Sensitivity = 1)
    traces['tau_group_count'] = n_taxis + np.random.laplace(0, 1.0 / COSTS['tau_group_count'])
    
    # Row Contribution: max entity contribution AFTER bounding to [0, C]
    raw_max = pings.max() if len(pings) > 0 else 0
    clipped_max = np.clip(raw_max, 0, C)
    traces['tau_row_contrib'] = clipped_max + np.random.laplace(0, C / COSTS['tau_row_contrib'])
    
    # Aggregation Composition: static query structure (zero cost)
    traces['tau_agg_comp'] = f"Sum(Clip(pings, {C}))"

    return traces

# ==========================================
# UTILITY CALCULATION 
# ==========================================
def calculate_utilities(cell_data, f_X, Y, traces):
    """
    U(tau_i) = Var(f(X) | Y) - Var(f(X) | Y, tau_i)       

    Uses grid-based discrete posterior inference over all plausible integer
    values of f(X) in [0, n_taxis * C].
    """
    
    n_taxis = len(cell_data)
    f_min = 0
    f_max = n_taxis * C

    f_grid = np.arange(f_min, f_max + 1, dtype=float)
    if len(f_grid) == 0:
        return {k: 0.0 for k in traces}, 0.0

    scale_query = C / EPSILON_QUERY  # = 10.0

    degenerate_threshold = 2.0 * scale_query   # 2 noise-scale units

    y_below_grid = Y < (f_min - degenerate_threshold)
    y_above_grid = Y > (f_max + degenerate_threshold)
    y_is_degenerate = y_below_grid or y_above_grid

    ## Before Metadata: compute posterior P(f(X) | Y) and its variance
    ## In the normal case, var_base = Var(f(X)|Y) || In the degenerate case, var_base = Var(Uniform[0, f_max])
    if y_is_degenerate:
        # Y is an extreme outlier — use a Uniform prior as the baseline.
        # This gives the honest variance reduction relative to "no information".
        # When Y is an extreme outlier, instead of using Y as evidence, the code assigns equal probability to every candidate value on the grid. 
        # E.g., For a cell with 9 taxis where f_grid = [0, 1, 2, ..., 45], every value gets probability 1/46.
        posterior_base = np.ones(len(f_grid)) / len(f_grid)
        mean_base = np.mean(f_grid)
        var_base  = np.var(f_grid)
        degenerate_flag = True
    else:
        # Normal case: Laplace likelihood on the grid
        # Laplace log-likelihood:
        log_ll_base = -np.abs(Y - f_grid) / scale_query
        # Numerical stability:
        log_ll_base -= np.max(log_ll_base)        
        # Convert to probabilities:  
        posterior_base = np.exp(log_ll_base)
        posterior_base /= posterior_base.sum()
        # Posterior mean and variance:
        mean_base = np.dot(posterior_base, f_grid)
        var_base  = np.dot(posterior_base, (f_grid - mean_base) ** 2)
        degenerate_flag = False

    # store log_posterior_base for combined updates below
    log_posterior_base = np.log(posterior_base + 1e-300)

    utilities = {}

    # ------------------------------------------------------------------
    # Utility of tau_noise
    # ------------------------------------------------------------------
    t_n        = traces['tau_noise']
    scale_eps  = C / COSTS['tau_noise']
    log_ll_noise = -np.abs(t_n - (Y - f_grid)) / scale_eps
    log_post = log_posterior_base + log_ll_noise
    log_post -= np.max(log_post)
    post = np.exp(log_post); post /= post.sum()
    mean_ = np.dot(post, f_grid)
    var_  = np.dot(post, (f_grid - mean_) ** 2)
    utilities['tau_noise'] = max(0.0, var_base - var_)

    # ------------------------------------------------------------------
    # Utility of tau_sens
    # ------------------------------------------------------------------
    t_s       = traces['tau_sens']
    scale_s   = C / COSTS['tau_sens']          
    if n_taxis > 1:
        # E[max-min | sum=v]: approximate spread of n_taxis values each ≈ v/n
        expected_sens = np.clip(f_grid / n_taxis, 0, C) * (1.0 - 1.0 / n_taxis)
    else:
        expected_sens = np.zeros_like(f_grid)
    log_ll_sens = -np.abs(t_s - expected_sens) / scale_s
    log_post = log_posterior_base + log_ll_sens
    log_post -= np.max(log_post)
    post = np.exp(log_post); post /= post.sum()
    mean_ = np.dot(post, f_grid)
    var_  = np.dot(post, (f_grid - mean_) ** 2)
    utilities['tau_sens'] = max(0.0, var_base - var_)

    # ------------------------------------------------------------------
    # Utility of tau_count
    # ------------------------------------------------------------------
    t_c         = traces['tau_count']
    scale_c     = 1.0 / COSTS['tau_count']       # = 2.5

    avg_ping    = f_grid / max(n_taxis, 1)        # ∈ [0, C]
    frac_clipped = np.clip(avg_ping / C, 0.0, 1.0) ** 2   # quadratic, ∈ [0,1]
    expected_clip = n_taxis * frac_clipped         # ∈ [0, n_taxis], varies with f

    log_ll_count = -np.abs(t_c - expected_clip) / scale_c
    log_post = log_posterior_base + log_ll_count
    log_post -= np.max(log_post)
    post = np.exp(log_post); post /= post.sum()
    mean_ = np.dot(post, f_grid)
    var_  = np.dot(post, (f_grid - mean_) ** 2)
    utilities['tau_count'] = max(0.0, var_base - var_)

    # ------------------------------------------------------------------
    # Utility tau_group_count
    # ------------------------------------------------------------------
    # Estimate: if the sum is f_grid, how many unique taxis do we expect?
    # We approximate by dividing the grid sum by the average ping rate of the cell.
    avg_ping_est = max(1.0, Y / max(n_taxis, 1)) 
    expected_groups = f_grid / avg_ping_est
    scale_g = 1.0 / COSTS['tau_group_count']
    
    log_ll_group = -np.abs(traces['tau_group_count'] - expected_groups) / scale_g
    log_post = log_posterior_base + log_ll_group
    log_post -= np.max(log_post)
    post = np.exp(log_post); post /= post.sum()
    mean_ = np.dot(post, f_grid)
    var_  = np.dot(post, (f_grid - mean_) ** 2)
    utilities['tau_group_count'] = max(0.0, var_base - var_)

    # ------------------------------------------------------------------
    # Utility tau_row_contrib
    # ------------------------------------------------------------------
    # Estimate: what is the expected max ping if the sum is f_grid?
    expected_max = np.clip(f_grid, 0, C) 
    scale_r = C / COSTS['tau_row_contrib']
    
    log_ll_row = -np.abs(traces['tau_row_contrib'] - expected_max) / scale_r
    log_post = log_posterior_base + log_ll_row
    log_post -= np.max(log_post)
    post = np.exp(log_post); post /= post.sum()
    mean_ = np.dot(post, f_grid)
    var_  = np.dot(post, (f_grid - mean_) ** 2)
    utilities['tau_row_contrib'] = max(0.0, var_base - var_)

    # ------------------------------------------------------------------
    # Utility tau_agg_comp
    # ------------------------------------------------------------------
    # This remains free config metadata (cost = 0.0).
    utilities['tau_agg_comp'] = 0.001

    return utilities, var_base, degenerate_flag


# ==========================================
# KNAPSACK OPTIMISATION
# ==========================================
def knapsack_optimize(utilities, costs, budget, scale_factor=100):
    """
    Optimal 0-1 Knapsack using Dynamic Programming.
    Time Complexity: O(n * W) where W is the scaled budget.
    """
    fields = list(utilities.keys())
    n = len(fields)
    
    # 1. Scale continuous epsilon floats to integers for the DP table
    W = int(budget * scale_factor)
    wt = [int(costs[f] * scale_factor) for f in fields]
    val = [utilities[f] for f in fields]
    
    # 2. Initialize DP table: dp[item_index][current_budget]
    dp = [[0.0 for _ in range(W + 1)] for _ in range(n + 1)]
    
    # 3. Build the DP table
    for i in range(1, n + 1):
        for w in range(W + 1):
            if wt[i-1] <= w:
                # Max of (including the item) OR (excluding the item)
                dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
                
    # 4. Traceback to find which specific fields were selected
    best_u = dp[n][W]
    best_subset = []
    w = W
    for i in range(n, 0, -1):
        # If the value comes from the row above, item was NOT included
        if dp[i][w] != dp[i-1][w]:
            best_subset.append(fields[i-1])
            w -= wt[i-1]  # Subtract weight to trace back the remaining budget
            
    return best_subset, best_u

def baseline_greedy(utilities, costs, budget):
    """Sorts by utility-to-cost ratio"""
    sorted_fields = sorted(utilities.keys(), 
                           key=lambda k: utilities[k] / (costs[k] + 1e-9), 
                           reverse=True)
    selected, current_cost, total_u = [], 0.0, 0.0
    for f in sorted_fields:
        if current_cost + costs[f] <= budget:
            selected.append(f)
            current_cost += costs[f]
            total_u += utilities[f]
    return selected, total_u

def baseline_random(utilities, costs, budget, trials=10):
    """Shuffles fields randomly, adding until budget runs out. Averages over N trials."""
    fields = list(utilities.keys())
    avg_u = 0.0
    for _ in range(trials):
        random.shuffle(fields)
        current_cost, trial_u = 0.0, 0.0
        for f in fields:
            if current_cost + costs[f] <= budget:
                current_cost += costs[f]
                trial_u += utilities[f]
        avg_u += trial_u
    return avg_u / trials # Return average utility across trials

def run_budget_sweep(grid_data, cell_totals, costs):
    """
    Tests the algorithms across a range of available privacy budgets
    to generate data for the Trade-off Line Graph.
    """
    print("\nRunning budget sweep for Privacy vs. Utility trade-off...")
    budgets_to_test = [0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5]
    sweep_results = []

    for _, row in cell_totals.iterrows():
        cx, cy = row['cell_x'], row['cell_y']
        cell_data = grid_data[(grid_data['cell_x'] == cx) & (grid_data['cell_y'] == cy)]
        
        # 1. Generate base mechanisms (only need to do this once per cell)
        f_X, Y, true_noise = run_base_mechanism(cell_data)
        traces = generate_metadata(cell_data, f_X, Y, true_noise)
        utilities, _, _ = calculate_utilities(cell_data, f_X, Y, traces)
        
        # 2. Test this cell against every budget level
        for b in budgets_to_test:
            _, opt_u = knapsack_optimize(utilities, costs, b)
            _, greedy_u = baseline_greedy(utilities, costs, b)
            random_u = baseline_random(utilities, costs, b, trials=5)
            
            sweep_results.append({
                'Budget': b,
                'Knapsack': opt_u,
                'Greedy': greedy_u,
                'Random': random_u
            })

    # Average the utilities across all cells for each budget level
    df_sweep = pd.DataFrame(sweep_results)
    summary_sweep = df_sweep.groupby('Budget').mean().reset_index()
    
    # Save to a new CSV for the plotting script
    summary_sweep.to_csv('budget_sweep_results.csv', index=False)
    print("Sweep complete! Saved to 'budget_sweep_results.csv'")

# ==========================================
# SCALABILITY & RUNTIME ANALYSIS
# ==========================================
import time

def run_scalability_test(filepath, costs, budget):
    """
    Tests computation time across increasing grid sizes.
    Directly compares Traditional DP vs. the complete Proposed Framework (Prov + Utility + Knapsack).
    """
    print("\n=== RUNNING SCALABILITY & RUNTIME ANALYSIS ===")
    grid_sizes = [5, 10, 15, 20]
    scale_results = []

    for g in grid_sizes:
        print(f"Testing {g}x{g} grid...")
        
        # Preprocessing (Not timed for algorithm benchmark)
        grid_data, cell_totals = preprocess_data(filepath, grid_size=g)

        t_traditional = 0.0
        t_proposed = 0.0

        for _, row in cell_totals.iterrows():
            cx, cy = row['cell_x'], row['cell_y']
            cell_data = grid_data[(grid_data['cell_x'] == cx) & (grid_data['cell_y'] == cy)]
            
            # 1. Traditional DP Time
            t0 = time.perf_counter()
            f_X, Y, true_noise = run_base_mechanism(cell_data)
            t_traditional += (time.perf_counter() - t0)
            
            # 2. Proposed Framework Time (Metadata + Utility + Knapsack)
            t1 = time.perf_counter()
            traces = generate_metadata(cell_data, f_X, Y, true_noise)
            utilities, _, _ = calculate_utilities(cell_data, f_X, Y, traces)
            knapsack_optimize(utilities, costs, budget)
            t_proposed += (time.perf_counter() - t1)

        scale_results.append({
            'Grid_Size': f"{g}x{g}",
            'Total_Cells': len(cell_totals),
            'Traditional_DP_sec': round(t_traditional, 5),
            'Proposed_Framework_sec': round(t_proposed, 5)
        })

    df_scale = pd.DataFrame(scale_results)
    df_scale.to_csv('scalability_results.csv', index=False)
    
    print("\nScalability Results:")
    print(df_scale.to_string(index=False))
    print("Saved -> 'scalability_results.csv'")
# ==========================================
# 7. MAIN PIPELINE
# ==========================================
def main():
    #filepath  = 'tax.csv'
    try:
        grid_data, cell_totals = preprocess_data('tax.csv', grid_size=20)
    except FileNotFoundError:
        print("Dataset 'tax.csv' not found. Please provide the file.")
        return
    #grid_data = preprocess_data(filepath, grid_size=10)

    results = []
    cell_dumper = CellDumper()    # Collects (U_i, Γ_i) for export

    print(f"Total Budget  (epsilon_total)  : {EPSILON_TOTAL}")
    print(f"Query Budget  (epsilon_query)  : {EPSILON_QUERY}")
    print(f"Metadata Budget (ep_remain)  : {EPSILON_REMAIN}")
    print(f"Field costs   (Gamma_i)        : {COSTS}")
    """
    print(f"Max affordable subset cost     : "
          f"{max(sum(COSTS[f] for f in s) for r in range(1,4) for s in combinations(COSTS,r) if sum(COSTS[f] for f in s)<=EPSILON_REMAIN):.2f}\n")
    """
    for _, row in cell_totals.iterrows():
        cx, cy, region = row['cell_x'], row['cell_y'], row['region']
        cell_data = grid_data[(grid_data['cell_x'] == cx) & (grid_data['cell_y'] == cy)]
        
        # 30 independent stochastic trials per cell
        trials = 30
        k_scores, g_scores, r_scores, base_vars, dp_outputs = [], [], [], [], []
        k_costs, k_counts, g_costs, g_counts, unconstrained_scores = [], [], [], [], []
        
        # To record the true sum (which doesn't change across trials)
        f_X, _, _ = run_base_mechanism(cell_data)
        
        degenerate_count = 0
        
        for _ in range(trials):
            # 1. Generate fresh noise and traces for this trial
            _, Y, true_noise = run_base_mechanism(cell_data)
            traces = generate_metadata(cell_data, f_X, Y, true_noise)
            utilities, var_base, degenerate = calculate_utilities(cell_data, f_X, Y, traces)
            
            # 2. Run optimizers
            opt_selected, opt_u = knapsack_optimize(utilities, COSTS, EPSILON_REMAIN)
            greedy_selected, greedy_u = baseline_greedy(utilities, COSTS, EPSILON_REMAIN)
            # Run random for 1 trial here since we are already inside a 30-trial loop
            rand_u = baseline_random(utilities, COSTS, EPSILON_REMAIN, trials=1)
            
            # 3. Store metrics for this trial
            k_scores.append(opt_u)
            g_scores.append(greedy_u)
            r_scores.append(rand_u)
            unconstrained_scores.append(sum(utilities.values()))
            base_vars.append(var_base)
            dp_outputs.append(Y)
            k_costs.append(sum(COSTS[f] for f in opt_selected))
            k_counts.append(len(opt_selected))
            g_costs.append(sum(COSTS[f] for f in greedy_selected))
            g_counts.append(len(greedy_selected))
            if degenerate: degenerate_count += 1

        results.append({
            'Region': region,
            'True_Sum': f_X,
            'DP_Output_Mean': np.mean(dp_outputs),
            'Var_Base_Mean': np.mean(base_vars),
            'Traditional_DP_Utility': 0.0,  # No metadata => no variance reduction by definition
            'Knapsack_Mean': np.mean(k_scores),
            'Knapsack_SEM': st.sem(k_scores),
            'Greedy_Mean': np.mean(g_scores),
            'Greedy_SEM': st.sem(g_scores),
            'Random_Mean': np.mean(r_scores),
            'Random_SEM': st.sem(r_scores),
            'Unconstrained_Mean': np.mean(unconstrained_scores),
            'Knapsack_Cost_Mean': np.mean(k_costs),
            'Knapsack_Fields_Mean': np.mean(k_counts),
            'Greedy_Cost_Mean': np.mean(g_costs),
            'Greedy_Fields_Mean': np.mean(g_counts),
            'Degenerate_Outlier_Count': degenerate_count 
        })
        cell_dumper.add_cell(
            cell_id=f"r{cx}c{cy}",
            region=region,
            epsilon_remain=EPSILON_REMAIN,
            utilities=utilities,    # already a dict in the right shape
            gammas=COSTS,           # your existing COSTS dict matches DEFAULT_GAMMAS
        )
    df_res = pd.DataFrame(results)
    
    # Print the aggregate comparison for Section 7
    print("\n=== BASELINE COMPARISON: AVERAGE EXPLANATORY UTILITY ===")
    
    # Updated summary 
    summary = df_res.groupby('Region').agg(
        Avg_True_Sum=('True_Sum', 'mean'),
        Avg_DP_Output=('DP_Output_Mean', 'mean'),
        Avg_Var_Base=('Var_Base_Mean', 'mean'),
        Traditional_DP_Utility=('Traditional_DP_Utility', 'mean'),  # always 0
        Knapsack_Utility=('Knapsack_Mean', 'mean'),
        Greedy_Utility=('Greedy_Mean', 'mean'),
        Random_Utility=('Random_Mean', 'mean'),
        Total_Cells=('Region', 'count'),
        Extreme_Outliers=('Degenerate_Outlier_Count', 'sum')
    ).reset_index()
    
    # Format the DP output to 2 decimal places for cleaner printing
    summary['Avg_DP_Output'] = summary['Avg_DP_Output'].round(2)
    summary['Avg_Var_Base'] = summary['Avg_Var_Base'].round(2)
    summary['Knapsack_Utility'] = summary['Knapsack_Utility'].round(2)
    summary['Greedy_Utility'] = summary['Greedy_Utility'].round(2)
    summary['Random_Utility'] = summary['Random_Utility'].round(2)
    
    print(summary.to_string(index=False))
    
    # Print table-ready format with ± SEM for the paper
    print("\n=== TABLE FOR PAPER (with ± SEM across cells per region) ===")
    for region in ['Rural', 'Suburban', 'Urban']:
        dr = df_res[df_res['Region'] == region]
        n = len(dr)
        print(f"{region:10s} & {n} "
              f"& {dr['True_Sum'].mean():.2f} "
              f"& {dr['DP_Output_Mean'].mean():.2f} "
              f"& {dr['Var_Base_Mean'].mean():.2f} "
              f"& 0 "
              f"& ${dr['Random_Mean'].mean():.2f} \\pm {st.sem(dr['Random_Mean']):.2f}$ "
              f"& ${dr['Greedy_Mean'].mean():.2f} \\pm {st.sem(dr['Greedy_Mean']):.2f}$ "
              f"& ${dr['Knapsack_Mean'].mean():.2f} \\pm {st.sem(dr['Knapsack_Mean']):.2f}$ \\\\")
    
    print("\n=== GLOBAL PERFORMANCE OVERALL ===")
    print(f"Traditional DP   : 0.0000 (no metadata released)")
    print(f"Random Baseline  : {df_res['Random_Mean'].mean():.4f}")
    print(f"Greedy Selector  : {df_res['Greedy_Mean'].mean():.4f}")
    print(f"Knapsack Selector: {df_res['Knapsack_Mean'].mean():.4f}")

    df_res.to_csv('experiment_results.csv', index=False)
    print("\nResults saved to 'experiment_results.csv'")
    cell_dumper.dump('data/cells_real.json')

    # Run the budget sweep to generate data for Plot 3
    run_budget_sweep(grid_data, cell_totals, COSTS)
    
    # Run the scalability test
    run_scalability_test('tax.csv', COSTS, EPSILON_REMAIN)

if __name__ == "__main__":
    main()