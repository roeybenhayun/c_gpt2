import numpy as np

# Define the same arrays as in C
c = np.array([
    [1.2, 2.3, 3.0, 4.9],
    [1.2, 2.3, 3.0, 4.0],
    [1.2, 2.3, 3.0, 4.9]
])  # 3x4

d = np.array([
    [0.3, 0.9, 1.1],
    [0.4, 0.9, 1.1],
    [0.5, 0.9, 1.1],
    [0.5, 0.9, 1.1]
])  # 4x3

# Matrix multiplication
result = np.dot(c, d)  # 3x3 matrix

# Optional: get the sum of all elements, if your C code does a total dot sum
dot_product_total = np.sum(result)

print("Dot product matrix:\n", result)
print(f"Total dot product sum: {dot_product_total:.17f}")
