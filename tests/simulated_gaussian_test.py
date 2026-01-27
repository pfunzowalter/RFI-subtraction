import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os

def generate_improved_rfi_diagnostic(dataset_file='training_results/astro_rfi_test_residuals.npz', num_examples=3):
# def generate_improved_rfi_diagnostic(dataset_file='astro_rfi_dataset.npz', num_examples=3):
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found.")
        return

    data = np.load(dataset_file)
    inputs = data['inputs']   
    targets = data['targets'] 
    
    indices = np.random.choice(len(inputs), num_examples, replace=False)

    for idx in indices:
        S = inputs[idx]
        T_ng = targets[idx]
        
        # --- FOCUS LOGIC ---
        # Instead of a global mask, we find the peak of the RFI (t0) 
        # and take a window of +/- 3 sigma (approx 150 samples based on your sigma=50)
        t0_detected = np.argmax(np.abs(T_ng))
        window_half_width = 100  # Captures the full Gaussian envelope of the RFI
        
        start = max(0, t0_detected - window_half_width)
        end = min(len(S), t0_detected + window_half_width)
        
        data_rfi_zone = S[start:end]
        # Clean zone is everything outside that specific burst window
        data_clean_zone = np.concatenate([S[:start], S[end:]])

        # Statistical Tests
        _, p_clean = stats.shapiro(data_clean_zone[:5000])
        _, p_rfi = stats.shapiro(data_rfi_zone)

        # Plotting
        fig = plt.figure(figsize=(15, 10))
        grid = fig.add_gridspec(2, 2)

        # 1. Top Panel: Time Series with Window Highlight
        ax_top = fig.add_subplot(grid[0, :])
        ax_top.plot(S, color='lightgray', label='Full Signal', alpha=0.5)
        ax_top.plot(range(start, end), S[start:end], color='crimson', label='Targeted RFI Window', linewidth=2)
        ax_top.axvline(t0_detected, color='black', linestyle='--', alpha=0.3, label='RFI Center (t0)')
        ax_top.set_title(f'Sample {idx}: Targeted RFI Analysis (Window: {start} to {end})')
        ax_top.legend(loc='upper right')

        # 2. Bottom Left: Histogram
        ax_hist = fig.add_subplot(grid[1, 0])
        sns.histplot(data_clean_zone, kde=True, ax=ax_hist, color='forestgreen', label='Clean Zone', stat="density", alpha=0.4)
        sns.histplot(data_rfi_zone, kde=True, ax=ax_hist, color='crimson', label='RFI Window', stat="density", alpha=0.4)
        ax_hist.set_title('Distribution Comparison')
        ax_hist.legend()

        # 3. Bottom Right: Q-Q Plot
        ax_qq = fig.add_subplot(grid[1, 1])
        stats.probplot(data_clean_zone, dist="norm", plot=ax_qq)
        ax_qq.get_lines()[0].set_markerfacecolor('forestgreen')
        ax_qq.get_lines()[0].set_label(f'Clean Sky (p={p_clean:.2e})')
        
        (osm, osr), _ = stats.probplot(data_rfi_zone, dist="norm", plot=None)
        ax_qq.scatter(osm, osr, color='crimson', s=15, label=f'RFI Window (p={p_rfi:.2e})', alpha=0.6)
        
        ax_qq.set_title('Normal Q-Q Plot (Targeted Window)')
        ax_qq.legend()

        plt.tight_layout()
        plt.savefig(f"targeted_rfi_test_{idx}.png", dpi=300)
        plt.close()
        print(f"Generated targeted diagnostic for sample {idx}")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    generate_improved_rfi_diagnostic()