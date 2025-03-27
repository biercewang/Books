import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of iterations and sample size
iterations = 1000
sample_size = 100

# Distributions to compare
distributions = {
    "Normal(μ=0, σ=1)": lambda n: np.random.normal(0, 1, n),
    "Exponential(λ=1)": lambda n: np.random.exponential(1, n),
    "Pareto(α=2)": lambda n: (np.random.pareto(2, n) + 1),  # Shift to have support [1, ∞)
}

# Re-plot with cumulative average to better visualize mean convergence

# Number of samples per trial (larger sample size for cumulative mean)
samples = 5000

# Recompute cumulative means for each distribution
plt.figure(figsize=(12, 8))

for label, dist_func in distributions.items():
    data = dist_func(samples)
    cumulative_means = np.cumsum(data) / np.arange(1, samples + 1)
    line = plt.plot(cumulative_means)[0]
    # 在每条线的末端添加标注
    plt.annotate(label, 
                xy=(samples-1, cumulative_means[-1]),
                xytext=(10, 0), 
                textcoords='offset points',
                va='center',
                color=line.get_color())

plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.annotate('Reference Line (mean=1)',
            xy=(samples-1, 1.0),
            xytext=(10, 0),
            textcoords='offset points',
            va='center',
            color='gray')

plt.xlabel("Sample Size")
plt.ylabel("Cumulative Sample Mean")
plt.title("Convergence of Sample Mean with Increasing Sample Size")
plt.grid(True)
plt.tight_layout()
plt.show()
