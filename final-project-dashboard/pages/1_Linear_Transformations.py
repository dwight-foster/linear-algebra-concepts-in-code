import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import compute_determinant

st.set_page_config(page_title="Linear Transformations Visualizer")

st.title("Linear Transformations Visualizer")

fig, ax = plt.subplots(figsize=(8, 8))

# Create unit circle


st.subheader("Transformation Matrix A")
col1, col2 = st.columns(2)

with col1:
    a11 = st.slider("a11", -3.0, 3.0, 1.0, 0.1)
    a12 = st.slider("a12", -3.0, 3.0, 0.0, 0.1)

with col2:
    a21 = st.slider("a21", -3.0, 3.0, 0.0, 0.1)
    a22 = st.slider("a22", -3.0, 3.0, 1.0, 0.1)

A = np.array([[a11, a12], [a21, a22]])
st.latex(r"A=\begin{bmatrix}" + f"{a11:.1f} & {a12:.1f} \\\\ {a21:.1f} & {a22:.1f}" + r"\end{bmatrix}")
det = compute_determinant(A)
st.latex(f"det(A) = {det:.2f}")
theta = np.linspace(0, 2*np.pi, 400)
circle = np.vstack((np.cos(theta), np.sin(theta)))   
transformed = A @ circle                              
x, V = np.linalg.eig(A)
ax.plot(circle[0], circle[1], 'k--', alpha=0.5, label='Unit circle')
ax.plot(transformed[0], transformed[1], 'b-', linewidth=2, label='A * circle')

for i in range(len(x)):
    eigenvector = V[:, i]
    ax.arrow(0, 0, eigenvector[0], eigenvector[1], head_width=0.15, head_length=0.1, fc='purple', ec='purple', alpha=0.7, linestyle=':', linewidth=2, label=f'Eigenvector {i+1} (λ={x[i]:.2f})')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title("Unit Circle and Its Image Under A")
ax.legend()

st.pyplot(fig)