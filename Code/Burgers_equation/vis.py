import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# Load the matrix data from file
rows, cols, values = [], [], []

with open("bij_python.txt", "r") as file:
    for line in file:
        r, c, v = map(float, line.split())
        rows.append(int(r))
        cols.append(int(c))
        values.append(v)

# Determine the matrix size
num_rows = max(rows) + 1
num_cols = max(cols) + 1

# Create a sparse matrix
sparse_matrix = coo_matrix((values, (rows, cols)), shape=(num_rows, num_cols))

# Convert to dense format for visualization (if it's not too large)
dense_matrix = sparse_matrix.toarray()

# Visualize with Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(dense_matrix, cmap="viridis", origin="upper", interpolation="nearest")
plt.colorbar(label="Matrix Values")
plt.title("Matrix Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()
