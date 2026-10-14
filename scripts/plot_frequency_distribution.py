import csv
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Paths
freq_csv = "c:/Users/Yashika/Desktop/AnurajSirJi/data/analysis/typology_parameter_frequency.csv"
output_img = "c:/Users/Yashika/Desktop/AnurajSirJi/outputs/rag_parameter_frequency/frequency_distribution.png"

# Load frequency percentages
percentages = []
with open(freq_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # frequency_percentage column may contain a trailing %
        fp = row.get('frequency_percentage') or row.get('frequency %')
        if not fp:
            continue
        # Remove non‑numeric characters
        m = re.search(r"([0-9]+(?:\.[0-9]*)?)", fp)
        if m:
            percentages.append(float(m.group(1)))

if not percentages:
    raise ValueError('No frequency percentages found in CSV')

# Histogram
plt.figure(figsize=(8, 5))
count, bins, ignored = plt.hist(percentages, bins=10, density=True, alpha=0.6, color='g', edgecolor='black')

# Fit a normal distribution
mu, std = norm.fit(percentages)

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = f"Distribution of Frequency Percentages (μ={mu:.1f}, σ={std:.1f})"
plt.title(title)
plt.xlabel('Frequency Percentage')
plt.ylabel('Density')
plt.grid(True)

# Save the figure
import os
os.makedirs(os.path.dirname(output_img), exist_ok=True)
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {output_img}")
