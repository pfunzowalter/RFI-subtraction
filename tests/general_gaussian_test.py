import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os

def simulate_and_save_tests(n_samples=1000, filename="gaussian_analysis.png"):
    """
    Simulates Gaussian noise and saves Visual Normality Tests to a file.
    """
    # 1. Simulate Gaussian Noise (μ=0, σ=1)
    np.random.seed(42)
    noise = np.random.normal(0, 1, n_samples)

    # 2. Setup Plot (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Test 1: Histogram + KDE ---
    sns.histplot(noise, kde=True, ax=axes[0], color='royalblue')
    axes[0].set_title('Histogram & KDE (Density)')
    axes[0].set_xlabel('Amplitude')
    axes[0].set_ylabel('Frequency')

    # --- Test 2: Q-Q Plot ---
    # Compares empirical quantiles to theoretical normal quantiles
    stats.probplot(noise, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot')
    axes[1].get_lines()[0].set_markerfacecolor('royalblue')
    axes[1].get_lines()[0].set_markeredgecolor('white')

    plt.suptitle(f"Normality Analysis: n={n_samples}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 3. Save the Plot
    plt.savefig(filename, dpi=300)
    print(f"Success: Visual tests saved to {os.path.abspath(filename)}")

    # 4. Brief Statistical Summary (printed to console)
    shapiro_stat, shapiro_p = stats.shapiro(noise)
    print(f"\nShapiro-Wilk Test p-value: {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print("Conclusion: Data appears Gaussian.")
    else:
        print("Conclusion: Data is non-Gaussian.")

if __name__ == "__main__":
    simulate_and_save_tests()