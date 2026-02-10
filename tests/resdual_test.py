import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

def run_normality_suite(data, alpha=0.05):
    """Calculates normality metrics for a given signal segment."""
    # Shapiro is sensitive, we cap at 5000 samples for stability
    _, p_shapiro = stats.shapiro(data[:5000])
    ad = stats.anderson(data, dist='norm')
    _, p_ks = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    _, p_dag = stats.normaltest(data)
    _, p_jb = stats.jarque_bera(data)
    
    sk = stats.skew(data)
    ku = stats.kurtosis(data, fisher=True)

    return {
        'shapiro': p_shapiro,
        'anderson_pass': ad.statistic < ad.critical_values[2], # 5% level
        'ks': p_ks,
        'dagostino': p_dag,
        'jarque_bera': p_jb,
        'skew': sk,
        'kurtosis': ku,
        'mean': np.mean(data),
        'std': np.std(data, ddof=1)
    }

def generate_rfi_diagnostic_v3(dataset_file='training_results/astro_rfi_test_residuals.npz', num_examples=3):
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found.")
        return

    data = np.load(dataset_file)
    inputs, predictions = data['inputs'], data['predictions']
    tg_hat = inputs - predictions 

    indices = np.random.choice(len(inputs), num_examples, replace=False)

    for idx in indices:
        t0 = np.argmax(np.abs(predictions[idx]))
        window_half = 100
        start, end = max(0, t0-window_half), min(len(inputs[idx]), t0+window_half)
        
        raw_zone = inputs[idx][start:end]
        clean_zone = tg_hat[idx][start:end]
        
        stats_raw = run_normality_suite(raw_zone)
        stats_clean = run_normality_suite(clean_zone)

        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. TOP PLOT: Time Domain Data
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.plot(inputs[idx], color='silver', alpha=0.5, label='Original Input (S)')
        ax_top.plot(tg_hat[idx], color='dodgerblue', alpha=0.8, label='Recovered Sky ($\hat{T}_g$)', lw=1)
        ax_top.axvspan(start, end, color='crimson', alpha=0.15, label='Statistical Analysis Zone')
        ax_top.set_title(f'Sample {idx}: Time-Domain RFI Mitigation Overview', fontsize=16, fontweight='bold')
        ax_top.legend(loc='upper right', ncol=3)

        # 2. Histogram + KDE
        ax1 = fig.add_subplot(gs[1, 0])
        sns.histplot(raw_zone, color='crimson', label='Raw', stat='density', alpha=0.3, ax=ax1)
        sns.histplot(clean_zone, color='royalblue', label='Cleaned', stat='density', alpha=0.5, ax=ax1)
        ax1.set_title('PDF: Before vs After')
        ax1.legend()

        # 3. Q-Q Plot
        ax2 = fig.add_subplot(gs[1, 1])
        (osm_r, osr_r), _ = stats.probplot(raw_zone, dist="norm")
        (osm_c, osr_c), _ = stats.probplot(clean_zone, dist="norm")
        ax2.scatter(osm_r, osr_r, color='crimson', s=12, alpha=0.3, label='Raw')
        ax2.scatter(osm_c, osr_c, color='royalblue', s=12, alpha=0.6, label='Cleaned')
        ax2.plot(osm_c, osm_c, 'k--', alpha=0.5)
        ax2.set_title('Normal Q-Q Plot')
        ax2.legend()

        # 4. ECDF
        ax3 = fig.add_subplot(gs[1, 2])
        sorted_c = np.sort(clean_zone)
        ecdf = np.arange(1, len(sorted_c)+1) / len(sorted_c)
        ax3.plot(sorted_c, ecdf, color='royalblue', lw=2, label='Cleaned')
        ax3.plot(sorted_c, stats.norm.cdf(sorted_c, stats_clean['mean'], stats_clean['std']), 'r--', label='Target Normal')
        ax3.set_title('ECDF (Recovered Signal)')
        ax3.legend()

        # 5. Results Table
        ax_table = fig.add_subplot(gs[2:, :])
        ax_table.axis('off')
        
        table_data = [
            ["Metric", "Raw RFI (Before)", "Recovered Sky (After)", "Ideal Target"],
            ["Shapiro-Wilk (p)", f"{stats_raw['shapiro']:.2e}", f"{stats_clean['shapiro']:.2e}", "> 0.05"],
            ["D'Agostino-P (p)", f"{stats_raw['dagostino']:.2e}", f"{stats_clean['dagostino']:.2e}", "> 0.05"],
            ["Jarque-Bera (p)", f"{stats_raw['jarque_bera']:.2e}", f"{stats_clean['jarque_bera']:.2e}", "> 0.05"],
            ["Skewness", f"{stats_raw['skew']:.4f}", f"{stats_clean['skew']:.4f}", "~ 0.00"],
            ["Excess Kurtosis", f"{stats_raw['kurtosis']:.4f}", f"{stats_clean['kurtosis']:.4f}", "~ 0.00"],
            ["Anderson-Darling", "REJECTED" if not stats_raw['anderson_pass'] else "PASS", 
                                "PASS" if stats_clean['anderson_pass'] else "REJECTED", "PASS"]
        ]
        
        table = ax_table.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3.0) 
        
        for j in range(4):
            table[(0, j)].set_text_props(weight='bold')
            table[(0, j)].set_facecolor('#eeeeee')
        
        # Dual-column color coding
        for i in range(1, len(table_data)):
            for col_idx in [1, 2]:
                cell = table[(i, col_idx)]
                val_str = table_data[i][col_idx]
                
                if i == 6: 
                    is_passing = (val_str == "PASS")
                elif i < 4:
                    is_passing = (float(val_str) > 0.05)
                else:
                    is_passing = (abs(float(val_str)) < 0.5)

                cell.set_facecolor('#d4edda' if is_passing else '#f8d7da')

        plt.suptitle(f'RFI Mitigation Diagnostic Report | Sample: {idx}', fontsize=20, y=0.98)
        save_name = f"rfi_diagnostic_report_{idx}.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    generate_rfi_diagnostic_v3()