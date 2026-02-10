import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

#-------------Resources for the tests used in this script-----------------------
# 1. Shapiro-Wilk Test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
# 2. Agostino-Pearson: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
# 3. Jarque-Bera: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html
# 4. Kurtosis: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
# 5. Skewness: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
# 6. AstroML: https://www.astroml.org/book_figures_1ed/chapter4/fig_anderson_darling.html

class NormalityAnalyzer:
    """
    A professional suite for performing and visualizing normality tests.
    """
    
    def __init__(self, data, alpha=0.05, label="Dataset"):
        self.data = np.array(data)
        self.alpha = alpha
        self.label = label
        self.results = {}
        self.stats_summary = {}

    def run_analysis(self):
        """Executes all statistical tests and calculates summaries."""
        self._calculate_basic_stats()
        self._test_shapiro()
        self._test_anderson()
        self._test_ks()
        self._test_dagostino()
        self._test_jarque_bera()
        self._test_moments()
        return self.results

    def _calculate_basic_stats(self):
        self.stats_summary = {
            'n': len(self.data),
            'mean': np.mean(self.data),
            'std': np.std(self.data, ddof=1),
            'min': np.min(self.data),
            'max': np.max(self.data)
        }

    def _test_shapiro(self):
        stat, p = stats.shapiro(self.data) # Ref 1
        self.results['Shapiro-Wilk'] = {'stat': stat, 'p': p, 'pass': p > self.alpha} 

    def _test_anderson(self):
        res = stats.anderson(self.data, dist='norm') # Ref 1
        # Using the 5% critical value (index 2)
        self.results['Anderson-Darling'] = {
            'stat': res.statistic, 
            'crit': res.critical_values[2], 
            'pass': res.statistic < res.critical_values[2],
            'all_crit': res.critical_values,
            'levels': res.significance_level
        }

    def _test_ks(self):
        # Kolmogorov-Smirnov against a normal distribution with sample mean/std
        stat, p = stats.kstest(self.data, 'norm', args=(self.stats_summary['mean'], self.stats_summary['std'])) # Ref 1
        self.results['Kolmogorov-Smirnov'] = {'stat': stat, 'p': p, 'pass': p > self.alpha}

    def _test_dagostino(self):
        stat, p = stats.normaltest(self.data) # Ref 2
        self.results["D'Agostino-Pearson"] = {'stat': stat, 'p': p, 'pass': p > self.alpha}

    def _test_jarque_bera(self):
        stat, p = stats.jarque_bera(self.data) # Ref 3
        self.results['Jarque-Bera'] = {'stat': stat, 'p': p, 'pass': p > self.alpha}

    def _test_moments(self):
        n = self.stats_summary['n']
        sk = stats.skew(self.data)
        se_sk = np.sqrt(6 / n)
        z_sk = sk / se_sk
        p_sk = 2 * (1 - stats.norm.cdf(np.abs(z_sk)))

        kur = stats.kurtosis(self.data, fisher=True)
        se_kur = np.sqrt(24 / n)
        z_kur = kur / se_kur
        p_kur = 2 * (1 - stats.norm.cdf(np.abs(z_kur)))

        self.results['Skewness'] = {'val': sk, 'p': p_sk, 'pass': p_sk > self.alpha} #Ref 4
        self.results['Kurtosis'] = {'val': kur, 'p': p_kur, 'pass': p_kur > self.alpha} #Ref 5

    def create_report_plot(self, filename=None):
        """Generates the multi-panel visualization."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Hist/KDE
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(self.data, kde=True, ax=ax1, color='royalblue', stat='density', alpha=0.6)
        x = np.linspace(self.stats_summary['min'], self.stats_summary['max'], 100)
        ax1.plot(x, stats.norm.pdf(x, self.stats_summary['mean'], self.stats_summary['std']), 'r--', label='Normal PDF')
        ax1.set_title('Distribution Overlay')
        ax1.legend()

        # 2. Q-Q Plot
        ax2 = fig.add_subplot(gs[0, 1])
        stats.probplot(self.data, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot')

        # 3. ECDF
        ax3 = fig.add_subplot(gs[0, 2])
        sorted_data = np.sort(self.data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax3.plot(sorted_data, ecdf, label='Data ECDF', lw=2)
        ax3.plot(sorted_data, stats.norm.cdf(sorted_data, self.stats_summary['mean'], self.stats_summary['std']), 'r--', label='Normal CDF')
        ax3.set_title('ECDF Comparison')
        ax3.legend()

        # 4. Box/Violin
        ax4 = fig.add_subplot(gs[1, 0])
        sns.boxplot(x=self.data, ax=ax4, color='lightblue')
        ax4.set_title('Box Plot')

        ax5 = fig.add_subplot(gs[1, 1])
        sns.violinplot(x=self.data, ax=ax5, color='lightgreen')
        ax5.set_title('Violin Plot')

        # 5. P-P Plot
        ax6 = fig.add_subplot(gs[1, 2])
        theoretical_probs = stats.norm.cdf(sorted_data, self.stats_summary['mean'], self.stats_summary['std'])
        ax6.scatter(theoretical_probs, ecdf, alpha=0.5, s=10)
        ax6.plot([0, 1], [0, 1], 'r--')
        ax6.set_title('P-P Plot')

        # 6. Table Summary
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        self._draw_table(ax7)

        plt.suptitle(f'Normality Analysis: {self.label} (n={self.stats_summary["n"]})', fontsize=18, y=0.95)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Report saved to {filename}")
        
        plt.show()

    def _draw_table(self, ax):
        table_data = [['Test Name', 'Statistic', 'p-value / Crit', 'Conclusion']]
        
        # Formal Tests
        for name in ['Shapiro-Wilk', "D'Agostino-Pearson", 'Jarque-Bera', 'Kolmogorov-Smirnov']:
            r = self.results[name]
            res_str = "✓ Normal" if r['pass'] else "✗ Non-Normal"
            table_data.append([name, f"{r['stat']:.4f}", f"{r['p']:.4e}", res_str])

        # Anderson (different format)
        ad = self.results['Anderson-Darling']
        ad_res = "✓ Normal" if ad['pass'] else "✗ Non-Normal"
        table_data.append(['Anderson-Darling', f"{ad['stat']:.4f}", f"Crit: {ad['crit']:.3f}", ad_res])

        # Moments
        sk, ku = self.results['Skewness'], self.results['Kurtosis']
        table_data.append(['Skewness', f"{sk['val']:.4f}", f"p: {sk['p']:.4e}", "Symmetric" if sk['pass'] else "Skewed"])
        table_data.append(['Kurtosis', f"{ku['val']:.4f}", f"p: {ku['p']:.4e}", "Mesokurtic" if ku['pass'] else "Heavy/Light"])

        table = ax.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Color coding
        for i in range(1, len(table_data)):
            cell = table[(i, 3)]
            if "✓" in table_data[i][3] or "Symmetric" in table_data[i][3]:
                cell.set_facecolor('#d4edda')
            elif "✗" in table_data[i][3] or "Skewed" in table_data[i][3]:
                cell.set_facecolor('#f8d7da')

### ------------------------EXECUTION BLOCK--------------------------------------------
if __name__ == "__main__":
    # Example 1: Generate Normal Data
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=10, size=1000)
    
    analyzer = NormalityAnalyzer(normal_data, label="Gaussian Simulation")
    analyzer.run_analysis()
    analyzer.create_report_plot("normal_report.png")

    # Example 2: Generate Non-Normal Data (Uniform)
    uniform_data = np.random.uniform(low=0, high=100, size=1000)
    
    analyzer_uni = NormalityAnalyzer(uniform_data, label="Uniform Simulation")
    analyzer_uni.run_analysis()
    analyzer_uni.create_report_plot("uniform_report.png")