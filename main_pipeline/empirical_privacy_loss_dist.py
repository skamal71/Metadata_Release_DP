"""
Empirical Privacy Loss Distribution
====================================
For each metadata field K_i, computes the pairwise log-likelihood ratio:
 
    |log Pr[K(x,m)=t] / Pr[K(x',m)=t]|
 
across sampled neighboring dataset pairs (x, x') and sampled outputs t.
Since all mechanisms are Laplace-based, the ratios have closed forms.
 
Output:
  1. One representative histogram per field (for the paper body)
  2. A summary bar chart with mean ± std across regions (for comparison)
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dp_metadata_pipeline import preprocess_data, C, EPSILON_QUERY, COSTS
 
GRID_SIZE = 10
N_SAMPLES_T = 500       # number of t samples per (x, x') pair
N_NEIGHBOR_PAIRS = 50   # number of neighboring pairs to test per cell
 
# ============================================================
# LOG RATIO FOR LAPLACE MECHANISMS
# ============================================================
# For K(x) = g(x) + Lap(b), the log-likelihood ratio is:
#   |log Pr[K(x)=t] / Pr[K(x')=t]| = (1/b) * ||t - g(x')| - |t - g(x)||
#
# This is bounded by |g(x) - g(x')| / b  (triangle inequality)
# which equals Gamma_i when |g(x) - g(x')| = sensitivity.
 
def compute_log_ratios_laplace(g_x, g_xprime, scale, n_samples):
    """
    Sample t from Lap(g_x, scale) and compute |log ratio| for each sample.
    Returns array of n_samples log-ratio values.
    """
    t_samples = np.random.laplace(loc=g_x, scale=scale, size=n_samples)
    diff = np.abs(t_samples - g_xprime) - np.abs(t_samples - g_x)
    log_ratios = np.abs(diff) / scale
    return log_ratios
 
 
def run_experiment(filepath='tax.csv'):
    print("Loading dataset...")
    grid_data, cell_totals = preprocess_data(filepath, grid_size=GRID_SIZE)
 
    # five data-aware fields with their properties
    fields = {
        'tau_noise': {
            'label': r'$\tau_{noise}$ (Noise residual)',
            'gamma': COSTS['tau_noise'],
            'scale': C / COSTS['tau_noise'],
            'sensitivity': C,  # Delta_f = C for clipped sum
        },
        'tau_sens': {
            'label': r'$\tau_{sens}$ (True sensitivity)',
            'gamma': COSTS['tau_sens'],
            'scale': C / COSTS['tau_sens'],
            'sensitivity': C,  # max-min can change by up to C
        },
        'tau_count': {
            'label': r'$\tau_{count}$ (Clipping count)',
            'gamma': COSTS['tau_count'],
            'scale': 1.0 / COSTS['tau_count'],
            'sensitivity': 1,  # count changes by at most 1
        },
        'tau_group_count': {
            'label': r'$\tau_{group}$ (Group count)',
            'gamma': COSTS['tau_group_count'],
            'scale': 1.0 / COSTS['tau_group_count'],
            'sensitivity': 1,
        },
        'tau_row_contrib': {
            'label': r'$\tau_{row}$ (Row contribution)',
            'gamma': COSTS['tau_row_contrib'],
            'scale': C / COSTS['tau_row_contrib'],
            'sensitivity': C,
        },
    }
 
    # Collect all results
    all_results = []  # per-field, per-region distributions
    summary_results = []  # mean/std per field per region
 
    for _, row in cell_totals.iterrows():
        cx, cy = row['cell_x'], row['cell_y']
        region = row['region']
        cell_data = grid_data[(grid_data['cell_x'] == cx) & (grid_data['cell_y'] == cy)]
        pings = cell_data['pings'].values
        n_taxis = len(cell_data)
 
        if n_taxis < 2:
            continue
 
        clipped = np.clip(pings, 0, C)
        f_X = clipped.sum()
 
        # base mechanism output 
        true_noise = np.random.laplace(0, C / EPSILON_QUERY)
        Y = f_X + true_noise
 
        # For each field, compute g(x) and simulate neighboring g(x')
        for field_name, field_info in fields.items():
            gamma = field_info['gamma']
            scale = field_info['scale']
            sens = field_info['sensitivity']
 
            # Compute g(x) for current dataset
            if field_name == 'tau_noise':
                g_x = Y - f_X  # = true_noise
            elif field_name == 'tau_sens':
                g_x = pings.max() - pings.min() if n_taxis > 1 else 0
            elif field_name == 'tau_count':
                g_x = (pings > C).sum()
            elif field_name == 'tau_group_count':
                g_x = n_taxis
            elif field_name == 'tau_row_contrib':
                g_x = np.clip(pings.max(), 0, C)

            field_log_ratios = []
            for _ in range(N_NEIGHBOR_PAIRS):
                delta = np.random.uniform(0, sens)
                sign = np.random.choice([-1, 1])
                g_xprime = g_x + sign * delta
 
                # Sample t values and compute log ratios
                ratios = compute_log_ratios_laplace(g_x, g_xprime, scale, N_SAMPLES_T)
                field_log_ratios.extend(ratios)
 
            field_log_ratios = np.array(field_log_ratios)
 
            all_results.append({
                'field': field_name,
                'field_label': field_info['label'],
                'region': region,
                'gamma': gamma,
                'log_ratios': field_log_ratios,
                'mean': field_log_ratios.mean(),
                'std': field_log_ratios.std(),
                'median': np.median(field_log_ratios),
                'p95': np.percentile(field_log_ratios, 95),
                'p99': np.percentile(field_log_ratios, 99),
                'max': field_log_ratios.max(),
                'violation_rate': (field_log_ratios > gamma).mean() * 100,
            })
 
            summary_results.append({
                'field': field_name,
                'region': region,
                'gamma': gamma,
                'mean_PL': field_log_ratios.mean(),
                'std_PL': field_log_ratios.std(),
                'median_PL': np.median(field_log_ratios),
                'p95_PL': np.percentile(field_log_ratios, 95),
                'max_PL': field_log_ratios.max(),
            })
 
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv('privacy_loss_distribution_summary.csv', index=False)
    print("Summary saved to 'privacy_loss_distribution_summary.csv'")
 
    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("EMPIRICAL PRIVACY LOSS DISTRIBUTION SUMMARY")
    print("=" * 70)
    agg = df_summary.groupby('field').agg(
        Gamma=('gamma', 'first'),
        Mean_PL=('mean_PL', 'mean'),
        Std_PL=('std_PL', 'mean'),
        Median_PL=('median_PL', 'mean'),
        P95_PL=('p95_PL', 'mean'),
        Max_PL=('max_PL', 'max'),
    ).round(4)
    print(agg.to_string())
 
    # ============================================================
    # PLOT 1: histogram for ONE field (tau_noise)
    # ============================================================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
 
    # Collect all log ratios for tau_noise across all cells
    noise_ratios = np.concatenate([
        r['log_ratios'] for r in all_results if r['field'] == 'tau_noise'
    ])
    gamma_noise = COSTS['tau_noise']
    mean_noise = noise_ratios.mean()
 
    fig, ax = plt.subplots(figsize=(8, 5))
 
    n_bins = 40
    bins = np.linspace(0, gamma_noise, n_bins + 1)
    ax.hist(noise_ratios, bins=bins, color='#5B9BD5', edgecolor='white',
            linewidth=0.3, alpha=0.85, label='Below bound')
 
    ax.axvline(gamma_noise, color='#C0392B', linewidth=2, linestyle='-',
               label=rf'Worst-case bound ($\Gamma$ = {gamma_noise})')
    ax.axvline(mean_noise, color='#F39C12', linewidth=1.5, linestyle='--',
               label=f'Mean: {mean_noise:.3f}')
 
    ax.set_xlabel(r'Pairwise log-likelihood ratio $\left|\log \frac{Pr[\mathcal{K}(x,m)=t]}{Pr[\mathcal{K}(x\prime,m)=t]}\right|$',
                  fontweight='bold')
    ax.set_ylabel('Count (number of (x, x\', t) tuples)', fontweight='bold')
    ax.set_title(r'Empirical privacy loss distribution: $\tau_{noise}$',
                 fontweight='bold', pad=15)
    ax.legend(frameon=True, fontsize=10)
 
    plt.tight_layout()
    plt.savefig('plot_PL_distribution_noise.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved -> 'plot_PL_distribution_noise.pdf'")
    plt.close()
 
    # ============================================================
    # PLOT 2: Representative histogram for ALL fields (grid of 5)
    # ============================================================
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
 
    for ax, field_name in zip(axes, fields.keys()):
        field_ratios = np.concatenate([
            r['log_ratios'] for r in all_results if r['field'] == field_name
        ])
        gamma = COSTS[field_name]
        mean_val = field_ratios.mean()
 
        # Bins end exactly at gamma
        n_bins = 30
        bins = np.linspace(0, gamma, n_bins + 1)
        ax.hist(field_ratios, bins=bins, color='#5B9BD5', edgecolor='white',
                linewidth=0.3, alpha=0.85)
 
        ax.axvline(gamma, color='#C0392B', linewidth=1.5, linestyle='-')
        ax.axvline(mean_val, color='#F39C12', linewidth=1.2, linestyle='--')
 
        short_name = field_name.replace('tau_', r'$\tau_{') + '}$'
        ax.set_title(short_name, fontweight='bold', fontsize=11)
        ax.set_xlabel('PL', fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel('Count', fontweight='bold')
 
        ax.text(0.95, 0.95, f'Mean: {mean_val:.2f}\n$\\Gamma$: {gamma}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
 
    plt.suptitle('Empirical privacy loss distribution across all metadata fields',
                 fontweight='bold', fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig('plot_PL_distribution_all_fields.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved -> 'plot_PL_distribution_all_fields.pdf'")
    plt.close()
 
    # ============================================================
    # PLOT 3: Mean PL vs Gamma per field, with error bars per region
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
 
    field_order = list(fields.keys())
    x_pos = np.arange(len(field_order))
    width = 0.2
    region_colors = {'Rural': '#2ECC71', 'Suburban': '#3498DB', 'Urban': '#E74C3C'}
 
    for i, region in enumerate(['Rural', 'Suburban', 'Urban']):
        dr = df_summary[df_summary['region'] == region]
        means = []
        stds = []
        for fn in field_order:
            dfn = dr[dr['field'] == fn]
            means.append(dfn['mean_PL'].mean())
            stds.append(dfn['std_PL'].mean())
        ax.bar(x_pos + i * width, means, width, yerr=stds, capsize=3,
               label=region, color=region_colors[region], edgecolor='black',
               linewidth=0.5, alpha=0.85)
 
    # Overlay Gamma values as horizontal markers
    for j, fn in enumerate(field_order):
        gamma = COSTS[fn]
        ax.plot([j - 0.1, j + 0.7], [gamma, gamma], 'k--', linewidth=1.0, alpha=0.6)
        ax.text(j + 0.75, gamma, f'$\\Gamma$={gamma}', fontsize=7, va='center', color='#666')
 
    short_labels = [r'$\tau_{noise}$', r'$\tau_{sens}$', r'$\tau_{count}$',
                    r'$\tau_{group}$', r'$\tau_{row}$']
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel('Mean empirical PL', fontweight='bold')
    ax.set_title('Mean empirical privacy loss vs. worst-case bound by region',
                 fontweight='bold', pad=15)
    ax.legend(frameon=True, title='Region')
 
    plt.tight_layout()
    plt.savefig('plot_PL_mean_vs_gamma.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved -> 'plot_PL_mean_vs_gamma.pdf'")
    plt.close()
 
    print("\nExperiment complete!")
 
 
if __name__ == "__main__":
    run_experiment()