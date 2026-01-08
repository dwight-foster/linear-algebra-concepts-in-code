import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

def random_nonparallel_vectors(rng, tol=1e-2):
    while True:
        u = rng.normal(size=3)
        v = rng.normal(size=3)
        if np.linalg.norm(u) < 1e-8 or np.linalg.norm(v) < 1e-8:
            continue
        if np.linalg.norm(np.cross(u, v)) > tol * np.linalg.norm(u) * np.linalg.norm(v):
            return u, v

# 1) Generate two random 3D vectors u, v
u, v = random_nonparallel_vectors(rng)

print("u =", u)
print("v =", v)

# 2) Generate example vectors on the span: w = a*u + b*v
#    (choose a few (a,b) pairs so you can see different combos)
coeffs = np.array([
    [ 1.0,  0.0],   # u
    [ 0.0,  1.0],   # v
    [ 1.0,  1.0],   # u + v
    [ 2.0, -1.0],   # 2u - v
    [-0.5,  1.5],   # -0.5u + 1.5v
    [ 0.7, -0.3],   # 0.7u - 0.3v
])

W = np.array([a * u + b * v for a, b in coeffs])

print("\nExample span vectors (w = a*u + b*v):")
for (a, b), w in zip(coeffs, W):
    print(f"a={a:>5}, b={b:>5}  ->  w={w}")

# 3) Plot just u, v, and a few w's on the span
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Two vectors u, v and example vectors on span{u, v}")

origin = np.zeros(3)


# Plot u and v as arrows with distinct colors
ax.quiver(0, 0, 0, u[0], u[1], u[2], color="red", label="u", linewidth=2)
ax.quiver(0, 0, 0, v[0], v[1], v[2], color="blue", label="v", linewidth=2)

# Plot the example span vectors as arrows (gray)
for i, w in enumerate(W):
    ax.quiver(0, 0, 0, w[0], w[1], w[2], color="gray", linewidth=1, label=f"w{i} = {coeffs[i,0]}u + {coeffs[i,1]}v")

# Make axes roughly comparable
pts = np.vstack([origin[None, :], u[None, :], v[None, :], W])
mins = pts.min(axis=0)
maxs = pts.max(axis=0)
cent = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2
ax.set_xlim(cent[0] - radius, cent[0] + radius)
ax.set_ylim(cent[1] - radius, cent[1] + radius)
ax.set_zlim(cent[2] - radius, cent[2] + radius)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Legend can get big; you can comment this out if it's cluttered
ax.legend(loc="upper left", fontsize=8)

plt.show()