import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from functions import compute_svd

st.set_page_config(page_title="PCA Analysis", layout="wide")
st.title("Principal Component Analysis (PCA)")
st.markdown("Explore dimensionality reduction using PCA")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is None:
    st.stop()
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Remove non-numerical columns
    df = df.select_dtypes(include=[np.number])
    
    st.write(df)
    # Convert dataframe to numpy array and standardize
    data = df.values
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
U, sigma, V = compute_svd(data)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sigma = np.diagonal(sigma)
# Plot first two principal components
axes[0].scatter(data @ V[:, 0], data @ V[:, 1], alpha=0.6)
arrow_scale = 2 * np.max(np.abs(data @ V[:, :2]))
arrow1 = axes[0].arrow(0, 0, V[0][0] * arrow_scale, V[0][1] * arrow_scale, 
                       width=0.05, color='red', head_width=0.3, length_includes_head=True, label='PC1 direction')
arrow2 = axes[0].arrow(0, 0, V[1][0] * arrow_scale, V[1][1] * arrow_scale, 
                       width=0.05, color='blue', head_width=0.3, length_includes_head=True, label='PC2 direction')
axes[0].legend(handles=[arrow1, arrow2])
axes[0].arrow(0, 0, V[1][0] * arrow_scale, V[1][1] * arrow_scale, 
                          width=0.05, color='blue', head_width=0.3, length_includes_head=True)
axes[0].set_xlabel(f"PC1 ({sigma[0]**2 / (sigma**2).sum() * 100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({sigma[1]**2 / (sigma**2).sum() * 100:.1f}%)")
axes[0].set_title("Data projected on first two principal components")
axes[0].grid(True, alpha=0.3)

# Plot variance explained by each component
variance_explained = (sigma**2 / (sigma**2).sum()) * 100
axes[1].bar(range(1, len(variance_explained) + 1), variance_explained)
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Variance Explained (%)")
axes[1].set_title("Variance explained by each component")
axes[1].grid(True, alpha=0.3, axis='y')

st.pyplot(fig)

st.markdown("### Reconstruct Data from Top k Principal Components")

k = st.slider("Select number of principal components (k)", min_value=1, max_value=min(data.shape), value=2)

if st.button("Reconstruct Data"):
    # TODO: Perform reconstruction using top k components
    reconstructed_data = U[:, :k] @ np.diag(sigma[:k]) @ V[:, :k].T
    st.write(pd.DataFrame(reconstructed_data, columns=df.columns))
    # Compare reconstructed data to original standardized data
    st.markdown("### Original data")
    st.write(pd.DataFrame(data, columns=df.columns))
