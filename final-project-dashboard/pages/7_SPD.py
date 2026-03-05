import streamlit as st
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from functions import generate_spd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quadratic Forms & Gradient Descent on SPD", layout="wide")
st.title("Quadratic Forms and Gradient Descent on Symmetric Positive Definite Matrices")

def to_bmatrix_latex(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return r"\begin{bmatrix}\end{bmatrix}"
    rows = [" & ".join(f"{val:g}" for val in row) for row in arr]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


matrix_size = st.slider("Matrix Size (rows/columns)", min_value=2, max_value=3, value=2)

if st.button("Generate SPD Matrix"):
    A = generate_spd(matrix_size)
    st.latex(r"\mathbf{A} = " + to_bmatrix_latex(A))
    b = np.random.randn(matrix_size)
    st.latex(r"\mathbf{b} = " + to_bmatrix_latex(b))
    st.session_state.b = b
    st.session_state.spd_matrix = A
    if matrix_size == 2:
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v = np.array([X[i, j], Y[i, j]])
                Z[i, j] = 0.5 * v @ A @ v - b @ v
        fig, ax = plt.subplots()
        contour = ax.contour(X, Y, Z, levels=15)
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$f(x) = \frac{1}{2} x^T A x - b^T x$")
        st.pyplot(fig)
    elif matrix_size == 3:
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v = np.array([X[i, j], Y[i, j], 0])
                Z[i, j] = 0.5 * v @ A @ v - b @ v
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$f(x, y)$")
        ax.set_title(r"$f(x) = \frac{1}{2} x^T A x - b^T x$")
        st.pyplot(fig)
    
  
if "spd_matrix" in st.session_state:
    A = st.session_state.spd_matrix
    b = st.session_state.b
    x, V = np.linalg.eig(A)
    lambda_min = min(x)
    lambda_max = max(x)
    k = lambda_max/lambda_min
    rec_step_size = 2 /(lambda_max + lambda_min)
    st.write(f"Recommended Step Size: {rec_step_size:.4f}")
    st.write(f"Condition Number (κ): {k:.4f}")
    st.write(f"λ_min: {lambda_min:.4f}")
    st.write(f"λ_max: {lambda_max:.4f}")
    step_size = st.slider("Gradient Descent Step Size", min_value=0.001, max_value=(2/lambda_max), value=rec_step_size, step=0.001)
    shrink_factor = max(1 - step_size*x)
    st.write(f"Shrink Factor: {shrink_factor:.4f}")
    st.session_state.step_size = step_size
    def gradient_descent(A, b, lr=0.1, iterations=10000):
        x = np.random.random(b.shape)
        xs = []
        for i in range(iterations):
            x -= lr * (A @ x - b)
            xs.append(x.copy())
        return x, np.array(xs)
    
    x, xs = gradient_descent(A, b, step_size)
    x_true = np.linalg.solve(A, b)
    fig, ax = plt.subplots()
    for i in range(len(xs[0])):
        ax.plot(xs[:, i], label=f"x_{i+1}")
        ax.axhline(y=x_true[i], linestyle='--', alpha=0.5, label=f"final_x_{i+1}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Gradient Descent Convergence")
    ax.legend()
    st.pyplot(fig)