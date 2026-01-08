import numpy as np
import matplotlib.pyplot as plt

basis = np.array([[1, 0], [0, 1]])
vector = np.random.uniform(-5, 5, 2)

def random_transformation():
    # Randomly choose rotation or shear
    if np.random.rand() < 0.5:
        # Rotation
        theta = np.random.uniform(0, 2 * np.pi)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    else:
        # Shear
        shear_factor = np.random.uniform(-2, 2)
        if np.random.rand() < 0.5:
            # Shear in x
            return np.array([
                [1, shear_factor],
                [0, 1]
            ])
        else:
            # Shear in y
            return np.array([
                [1, 0],
                [shear_factor, 1]
            ])

# Randomly choose two transformations
trans1 = random_transformation()
trans2 = random_transformation()

complete_transform = trans2 @ trans1
basis_transform = complete_transform @ basis
vector_transform = basis_transform @ vector

# Basis and vector after the first transformation (trans1)
mid_basis = trans1 @ basis
mid_vector = mid_basis @ vector

print("Basis:")
print(basis)
print("\nOriginal Vector:")
print(vector)
print("\nTransformation 1 (trans1):")
print(trans1)
print("\nTransformation 2 (trans2):")
print(trans2)
print("\nTransformed Basis:")
print(basis_transform)
print("\nTransformed Vector:")
print(vector_transform)

print("\nAfter first transformation (trans1) - Basis:")
print(mid_basis)
print("\nAfter first transformation (trans1) - Vector:")
print(mid_vector)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Two transformations using matrix multiplication")
ax.set_aspect('equal')  # keep x/y scales equal so arrows show correctly

quiver_opts = dict(angles='xy', scale_units='xy', scale=1, pivot='tail', width=0.008, headwidth=6, headlength=8)

i = basis[:, 0]
j = basis[:, 1]
ax.quiver(0, 0, i[0], i[1], color="r", label="i_hat (original)", **quiver_opts)
ax.quiver(0, 0, j[0], j[1], color="g", label="j_hat (original)", **quiver_opts)
ax.quiver(0, 0, vector[0], vector[1], color="b", label="vector (original)", **quiver_opts)

# Transformed basis vectors
trans_i = basis_transform[:, 0]
trans_j = basis_transform[:, 1]
ax.quiver(0, 0, trans_i[0], trans_i[1], color="m", label="i_hat (transformed)", **quiver_opts)
ax.quiver(0, 0, trans_j[0], trans_j[1], color="c", label="j_hat (transformed)", **quiver_opts)

# Transformed vector
ax.quiver(0, 0, vector_transform[0], vector_transform[1], color="y", label="vector (transformed)", **quiver_opts)

ax.legend()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.grid(True)
plt.show()
