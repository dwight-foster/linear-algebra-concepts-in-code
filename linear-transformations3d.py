import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

# Basis matrix: columns are e1, e2, e3
basis = np.eye(3)
orig_vector = np.random.randint(-5, 6, 3)
A = np.random.randint(-3, 4, (3, 3))
transformed_basis = A @ basis
transformed_vector = A @ orig_vector

print("Original vector:", orig_vector)
print("Transformation Matrix:\n", A)
print("Transformed vector (A @ orig_vector):", transformed_vector)
print("Transformed basis:\n", transformed_basis)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Original and Transformed Vectors and Basis (3D)")

# Plot arrows from origin
kwargs = dict(length=1, normalize=False)
ax.quiver(0, 0, 0, basis[0,0], basis[1,0], basis[2,0], color='r', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)
ax.quiver(0, 0, 0, basis[0,1], basis[1,1], basis[2,1], color='g', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)
ax.quiver(0, 0, 0, basis[0,2], basis[1,2], basis[2,2], color='b', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)

ax.quiver(0, 0, 0, orig_vector[0], orig_vector[1], orig_vector[2], color='k', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)

ax.quiver(0, 0, 0, transformed_basis[0,0], transformed_basis[1,0], transformed_basis[2,0], color='orange', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)
ax.quiver(0, 0, 0, transformed_basis[0,1], transformed_basis[1,1], transformed_basis[2,1], color='purple', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)
ax.quiver(0, 0, 0, transformed_basis[0,2], transformed_basis[1,2], transformed_basis[2,2], color='cyan', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)

ax.quiver(0, 0, 0, transformed_vector[0], transformed_vector[1], transformed_vector[2], color='magenta', linewidth=1.5, arrow_length_ratio=0.1, **kwargs)

# auto-expand axes to include all endpoints
all_pts = np.vstack([basis.T, orig_vector, transformed_basis.T, transformed_vector])
mins = all_pts.min(axis=0) - 5
maxs = all_pts.max(axis=0) + 5
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Create legend proxies
proxies = [
	mlines.Line2D([], [], color='r', label='e1'),
	mlines.Line2D([], [], color='g', label='e2'),
	mlines.Line2D([], [], color='b', label='e3'),
	mlines.Line2D([], [], color='k', label='Original vector'),
	mlines.Line2D([], [], color='orange', label='A*e1'),
	mlines.Line2D([], [], color='purple', label='A*e2'),
	mlines.Line2D([], [], color='cyan', label='A*e3'),
	mlines.Line2D([], [], color='magenta', label='A*vector'),
]
ax.legend(handles=proxies, loc='upper left')

plt.show()

