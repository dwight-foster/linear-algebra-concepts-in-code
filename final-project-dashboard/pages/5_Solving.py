import streamlit as st
import numpy as np
from scipy.linalg import solve, hilbert
from numpy.linalg import cond, inv
from functions import compute_solution
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solving Ill-Conditioned Systems", layout="wide")

st.title("Solving Ill-Conditioned Linear Systems")
st.markdown("""
Explore how ill-conditioned matrices cause numerical instability when solving Ax=b.
See why using matrix inverse is dangerous compared to direct solvers.
""")


st.header("Matrix Configuration")
matrix_type = st.radio("Choose matrix type:", ["Hilbert", "Nearly Dependent Columns"])
matrix_size = st.slider("Matrix size (n×n):", 3, 12, 6)
compute_button = st.button("Compute Solution")

if compute_button:
    if matrix_type == "Hilbert":
        A = hilbert(matrix_size)
    else: 
        A = np.random.randn(matrix_size, matrix_size)
        A[:, -1] = A[:, -2] + 1e-6 * np.random.randn(matrix_size)

    x_solution = np.random.random(matrix_size)
    b = A @ x_solution

    st.session_state.A = A
    st.session_state.x_solution = x_solution
    st.session_state.b = b
    st.session_state.pertrub_random = np.random.randn(matrix_size)


if "A" in st.session_state:
    
    A = st.session_state.A
    b = st.session_state.b
    x_solution = st.session_state.x_solution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Using Matrix Inverse")
        try:
            x_inverse = inv(A) @ b
            error_inverse = np.linalg.norm(x_inverse - x_solution)
            st.metric("Solution Error", f"{error_inverse:.2e}")
            st.write(f"Condition number: {cond(A):.2e}")
        except:
            st.error("Failed to compute inverse")

    with col2:
        st.subheader("Using Direct Solver")
        x_direct = np.linalg.solve(A, b)
        error_direct = np.linalg.norm(x_direct - x_solution)
        st.metric("Solution Error", f"{error_direct:.2e}")
        st.write(f"Condition number: {cond(A):.2e}")
    st.header("Solution Comparison")
    solution_difference = np.linalg.norm(x_inverse - x_direct)
    st.metric("Difference Between Methods", f"{solution_difference:.2e}")

    st.header("Perturbation Analysis")
    perturbation_scale = st.slider("Perturbation scale for b:", 10e-13, 10e-2, 10e-13, format="%.2e")
    b_perturbed = b + perturbation_scale * st.session_state.pertrub_random

    st.write("### Results with Perturbed b:")
    col1, col2 = st.columns(2)

    with col1:
        x_inverse_pert = inv(A) @ b_perturbed
        error_inverse_pert = np.linalg.norm(x_inverse_pert - x_solution)
        st.metric("Inverse Method Error", f"{error_inverse_pert:.2e}")

    with col2:
        x_direct_pert = np.linalg.solve(A, b_perturbed)
        error_direct_pert = np.linalg.norm(x_direct_pert - x_solution)
        st.metric("Direct Solver Error", f"{error_direct_pert:.2e}")