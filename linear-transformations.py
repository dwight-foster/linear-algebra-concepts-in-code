import numpy as np
import matplotlib.pyplot as plt

# Basis matrix: columns are i_hat and j_hat
basis = np.array([[1, 0], [0, 1]])
orig_vector = np.random.randint(-5, 5, 2)
A = np.random.randint(-3, 3, (2, 2))
transformed_basis = A @ basis
transformed_vector = basis @ orig_vector # i_hat * origi_vector[0] + j_hat * orig_vector[1]

print("Original vector:", orig_vector)
print("Transformation Matrix:", A)
print("Transformed vector (basis @ orig_vector):", transformed_vector)
print("Transformed vector (A @ orig_vector):", A @ orig_vector)
print("Transformed basis:", transformed_basis)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Original vector and transformed vector and basis")

origin = np.zeros(2)

# Plot basis vectors correctly
# use columns explicitly and disable quiver auto-scaling so lengths are in data units
i = basis[:, 0]
j = basis[:, 1]
tb = transformed_basis
tv = A @ orig_vector

kwargs = dict(angles='xy', scale_units='xy', scale=1, width=0.005)

ax.quiver(0, 0, i[0], i[1], color='r', label='i_hat', **kwargs)
ax.quiver(0, 0, j[0], j[1], color='g', label='j_hat', **kwargs)
ax.quiver(0, 0, orig_vector[0], orig_vector[1], color='b', label='Original vector', **kwargs)
ax.quiver(0, 0, tb[0, 0], tb[1, 0], color='orange', label='A*i_hat', **kwargs)
ax.quiver(0, 0, tb[0, 1], tb[1, 1], color='purple', label='A*j_hat', **kwargs)
ax.quiver(0, 0, tv[0], tv[1], color='cyan', label='A*vector', **kwargs)

# auto-expand axes to include all plotted endpoints
all_pts = np.vstack([i, j, orig_vector, tb[:, 0], tb[:, 1], tv])
mins = all_pts.min(axis=0) - 5
maxs = all_pts.max(axis=0) + 5
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.legend()

plt.show()