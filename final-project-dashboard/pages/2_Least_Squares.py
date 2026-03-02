import streamlit as st
import numpy as np
from functions import least_squares
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Least Squares",
    page_icon="📊",
    layout="wide"
)

st.title("Least Squares Method")


def to_bmatrix_latex(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return r"\begin{bmatrix}\end{bmatrix}"
    rows = [" & ".join(f"{val:g}" for val in row) for row in arr]
    return r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"


col1, col2 = st.columns(2)
x, y = np.array([]), np.array([])
with col1:
    x_input = st.text_input("Enter x values (comma separated):")
    if x_input:
        x = np.array([float(val.strip()) for val in x_input.split(",")])

with col2:
    y_input = st.text_input("Enter y values (comma separated):")
    if y_input:
        y = np.array([float(val.strip()) for val in y_input.split(",")])

inputs_provided = bool(x_input and y_input)
valid_lengths = len(x) == len(y)

if inputs_provided and not valid_lengths:
    st.error("Error: x and y values must have the same length")

if inputs_provided and valid_lengths:
    # Construct the design matrix A and vector b for least squares
    A = np.column_stack([np.ones(len(x)), x])
    b = y

    # Display A and b in LaTeX form
    st.subheader("Least Squares Setup")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Matrix A:**")
        st.latex(r"A = " + to_bmatrix_latex(A))

    with col2:
        st.write("**Vector b:**")
        st.latex(r"b = " + to_bmatrix_latex(b))

    x_hat = least_squares(A, b)

    st.subheader("Solution")
    st.latex(r"\hat{x} = " + to_bmatrix_latex(x_hat))
    st.latex(rf"\text{{Fitted line: }} y = {x_hat[0]:.4f} + {x_hat[1]:.4f}x")

    p = A @ x_hat
    e = b - p

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Projection (p):**")
        st.latex(r"p = " + to_bmatrix_latex(p))

    with col2:
        st.write("**Error (e):**")
        st.latex(r"e = " + to_bmatrix_latex(e))

    st.subheader("Visualization")
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data points", color="blue")
    ax.plot(x, p, label="Fitted line", color="red")
    ax.vlines(x, p, y, colors="gray", linestyles="--", linewidth=1.5, label="Residuals")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Verification")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**A.T @ e (should be ≈ 0):**")
        verification1 = np.where(np.isclose(A.T @ e, 0), 0, A.T @ e)
        st.latex(r"A^T e = " + to_bmatrix_latex(verification1))

    with col2:
        st.write("**P² = P (should be true):**")
        P = A @ np.linalg.inv(A.T @ A) @ A.T
        P_squared = P @ P
        st.latex(r"P^2 = " + to_bmatrix_latex(P_squared))
        st.latex(r"P = " + to_bmatrix_latex(P))
        st.write(f"Match: {np.allclose(P_squared, P)}")