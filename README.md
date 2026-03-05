# Linear Algebra Implementations (MIT OCW / Strang)

This repository is a hands-on implementation project based on Gilbert Strang's linear algebra lectures from MIT OpenCourseWare (18.06SC).  
The goal is to demonstrate practical understanding by coding core concepts from scratch and visualizing them.

## What This Repo Shows

- Ability to implement core linear algebra operations with NumPy/SciPy
- Ability to connect theory to computation (e.g., elimination, subspaces, least squares, SVD, eigen methods)
- Ability to build visual and interactive tools for concepts (matplotlib + Streamlit)

## Concepts Implemented

- Vectors, linear combinations, span, and basis
- Linear transformations in 2D and 3D
- Matrix multiplication as composition of transformations
- Determinants
- Inverse, column space, and null space
- Solving linear systems with row reduction / RREF
- Least squares via normal equations
- PCA via SVD
- Markov matrices and steady-state behavior
- Gram-Schmidt orthogonalization
- Symmetric positive definite (SPD) systems and gradient descent

## Project Structure

- `linear-transformations.py`: 2D transformation and basis visualization
- `linear-transformations3d.py`: 3D transformation and basis visualization
- `matmul-composition.py`: composition of transformations
- `3d-span.py`: span examples in 3D
- `determinants.py`: determinant computations and checks
- `inverse_column_null.py`: inverse, column space, and null space utilities
- `final-project-dashboard/`: Streamlit dashboard with interactive concept demos
- `strang_practice.ipynb`: notebook experiments and practice

## Dashboard (Interactive Final Project)

The Streamlit app includes pages for:

- Linear transformations
- Least squares
- PCA
- Solving systems
- Markov matrices
- Gram-Schmidt
- SPD + optimization intuition

Run it with:

```bash
cd final-project-dashboard
streamlit run main.py
```

## Run Script Examples

From the repo root:

```bash
python linear-transformations.py
python matmul-composition.py
python determinants.py
```

## Tech Stack

- Python
- NumPy
- SciPy
- Matplotlib
- Streamlit

## Source Material

- MIT OpenCourseWare 18.06SC: Linear Algebra (Gilbert Strang)  
  https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/

---

Author: Edward Dwight Foster
