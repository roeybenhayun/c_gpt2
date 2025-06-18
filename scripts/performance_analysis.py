import json
import matplotlib.pyplot as plt
import numpy as np

# File paths
files = {
    "gpt2_small": "../logs/gpt2_small_performance.json",
    "gpt2_medium": "../logs/gpt2_medium_performance.json",
    "gpt2_large": "../logs/gpt2_large_performance.json"
}

plt.figure(figsize=(10, 6))

# Process each file
for model_name, file_path in files.items():
    with open(file_path) as f:
        data = json.load(f)

    context_lengths = []
    chunk_times = []

    for chunk in data["chunks"]:
        context_lengths.append(chunk["total_context"])
        chunk_times.append(chunk["chunk_seconds"])

    # Convert to numpy arrays
    x = np.array(context_lengths)
    y = np.array(chunk_times)

    # Fit a quadratic curve
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    # Plot
    plt.plot(x, y, 'o', label=f'{model_name} data')
    plt.plot(x_fit, y_fit, '--', label=f'{model_name} fit')

plt.xlabel("Total Context Length")
plt.ylabel("Time for Last 32 Tokens (s)")
plt.title("Time to Generate Last 32 Tokens vs. Context Length")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
