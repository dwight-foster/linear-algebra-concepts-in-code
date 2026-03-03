import streamlit as st
import pandas as pd
import numpy as np
import graphviz

st.title("Markov Matrices and Directed Graphs")

st.markdown("""
Welcome to the Markov Matrices page!  
Here, you can create a directed graph, generate its corresponding Markov matrix, and explore its stable state using eigenvalues.
""")

# TODOs for Markov + Eigenvector Application

# 1. Allow user to create a small directed graph:
#    - Option to manually add nodes and edges.
#    - Option to select from preset graphs (e.g., 3-node cycle, star, etc.).

# 2. Build the transition (Markov) matrix from the graph:
#    - Compute transition probabilities for each node.
#    - Add a damping factor (e.g., for PageRank-style random jumps).

# 3. Show the stationary distribution:
#    - Compute the dominant eigenvector of the transition matrix.
#    - Display the stationary distribution (steady-state vector).

# 4. Visualize the graph and matrix:
#    - Show the directed graph.
#    - Display the transition matrix.
#    - Show the stationary distribution.

# 5. Add explanations and interactive controls:
#    - Sliders/inputs for damping factor.
#    - Step-by-step math explanations (optional).

# Implementation steps:
# - Build UI for graph creation/preset selection.
# - Generate transition matrix with damping.
# - Compute and display stationary distribution.
# - Visualize results.
# --- Scaffolding for Markov Matrix Dashboard ---

# Initialize session state for nodes and edges
if "nodes" not in st.session_state:
    st.session_state["nodes"] = ["A"]
    st.session_state['mapping'] = {"A": 0}
    st.session_state['idx'] = 1
if "edges" not in st.session_state:
    st.session_state["edges"] = {}

st.subheader("Add Edge to Graph")
start_node = st.selectbox("Start Node", st.session_state["nodes"])
end_node = st.text_input("End Node (new or existing)")

if st.button("Add Edge"):
    if end_node:
        if end_node not in st.session_state["nodes"]:
            st.session_state["nodes"].append(end_node)
            st.session_state['mapping'][end_node] = st.session_state['idx']
            st.session_state['idx'] += 1
        st.session_state["edges"].setdefault(start_node, set()).add(end_node)

# Visualize the directed graph using graphviz
dot = graphviz.Digraph()
for node in st.session_state["nodes"]:
    dot.node(node)
for start_node in st.session_state["edges"].keys():
    for end_node in st.session_state['edges'][start_node]:
        dot.edge(start_node, end_node)

st.graphviz_chart(dot)


transition_matrix = np.zeros((len(st.session_state['nodes']), len(st.session_state['nodes'])))
for i in range(len(st.session_state['nodes'])):
    node = st.session_state['nodes'][i]
    connections = st.session_state['edges'].get(node, [])
    for end_node in connections:
        idx = st.session_state['mapping'][end_node]
        transition_matrix[i][idx] = 1/len(connections)

damping = 0.85
transition_matrix = damping * transition_matrix + ((1-damping)/transition_matrix.shape[0])

print(transition_matrix)

