import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

st.set_page_config(page_title="Gram-Schmidt & QR", layout="wide")

st.title("Gram-Schmidt & QR Orthogonality Lab")

st.markdown("""
This page explores the Gram-Schmidt orthogonalization process and its connection to QR decomposition.
""")

st.divider()

# Sidebar controls
st.header("Configuration")
matrix_size = st.slider("Number of rows (n)", min_value=5, max_value=50, value=20)
num_cols = st.slider("Number of columns (m)", min_value=2, max_value=min(matrix_size, 15), value=5)
random_seed = st.checkbox("Use fixed random seed", value=True)

if st.button("Generate Matrix"):

    if random_seed:
        np.random.seed(42)

    # Generate random matrix
    A = np.random.randn(matrix_size, num_cols)
    # Display the generated matrix
    st.subheader("Generated Matrix A")
    st.dataframe(pd.DataFrame(A, columns=[f"Col {i+1}" for i in range(num_cols)]))

    # calculate Q using GS
    tab1, tab2, tab3 = st.tabs(["Classical Gram-Schmidt", "Modified Gram-Schmidt", "NumPy QR"])
    
    with tab1:
        st.subheader("Classical Gram-Schmidt")
        Q = []
        for i in range(A.shape[1]):
            v = A[:, i].copy()
            for q in Q:
                v -= np.dot(q, v) * q
            Q.append(v/np.linalg.norm(v))
        Q = np.column_stack(Q)
        st.dataframe(pd.DataFrame(Q))

        st.subheader("Q$^T$Q - I (should be ~0)")
        result = Q.T @ Q - np.eye(num_cols)
        st.latex(rf"Q^T Q - I = {np.linalg.norm(result):.2e}")

        # Calculate R from A = QR, so R = Q^T @ A
        R = Q.T @ A
        st.subheader("Reconstruction Error (||A - QR||)")
        reconstruction_error = np.linalg.norm(A - Q @ R)
        st.latex(rf"||A - QR|| = {reconstruction_error:.2e}")
    
    with tab2:
        st.subheader("Modified Gram-Schmidt")
        # TODO: Add modified Gram-Schmidt implementation
        pass
    
    with tab3:
        st.subheader("NumPy QR")
        # TODO: Add NumPy QR implementation
        pass