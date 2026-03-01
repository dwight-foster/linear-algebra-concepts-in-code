import streamlit as st

st.set_page_config(page_title="Final Project")

st.write("# Welcome to My Linear Algebra Final Project")

st.sidebar.success("Select a page.")

st.markdown(
    """
    This is the streamlit dashboard for my linear algebra final project.
    The class I took was from MIT OpenCourseWare [18.06SC](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)
    The pages are as follows:
    - Linear Transformations Visualizer: Visualize a linear transformation on a circle by choosing transformation matrix
    - Least squares: User enters data and can see the least squares projection on the data
    - PCA: User can upload data and see the PCA and reconstruction from SVD
    - Markov Matrices: User can create a directed graph and generate a markov matrix and stable state using eigenvalues
    - Solving: Generate an ill-conditioned matrix and show different solving methods with different results
    - Orthogonality: Using random columns finds the orthogonal matrix using Graham-Schmidt
    - SPD: Generate an SPD and run gradient descent with it to find the minimum and compare to actual minimum via elimination
    - Convolution: Creates a 1D blur kernel and shows how it blurs a signal 
    """
)