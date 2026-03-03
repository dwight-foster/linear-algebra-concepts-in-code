import streamlit as st
import pandas as pd
import numpy as np
import graphviz

st.title("Markov Matrices and Directed Graphs")

st.markdown("""
Welcome to the Markov Matrices page!  
Here, you can create a directed graph, generate its corresponding Markov matrix, and explore its stable state using eigenvalues.
""")

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

dot = graphviz.Digraph()
for node in st.session_state["nodes"]:
    dot.node(node)
for start_node in st.session_state["edges"].keys():
    for end_node in st.session_state['edges'][start_node]:
        dot.edge(start_node, end_node)

st.graphviz_chart(dot)

st.subheader("Compute Transition Matrix")
has_nodes = len(st.session_state["nodes"]) > 0

if st.button("Compute Transition Matrix", disabled=not has_nodes):

    if not has_nodes:
        st.error("Add at least one node to continue.")
    else:
        transition_matrix = np.zeros((len(st.session_state['nodes']), len(st.session_state['nodes'])))
        st.session_state["transition_matrix"] = transition_matrix
        st.session_state["nodes_snapshot"] = list(st.session_state["nodes"])  
        for i in range(len(st.session_state['nodes'])):
            node = st.session_state['nodes'][i]
            connections = st.session_state['edges'].get(node, st.session_state['nodes'])

            for end_node in connections:
                idx = st.session_state['mapping'][end_node]
                transition_matrix[idx][i] = 1/len(connections)

        damping = 0.85
        transition_matrix = damping * transition_matrix + ((1-damping)/transition_matrix.shape[0])

        st.subheader("Transition Matrix")
        matrix_latex = "\\begin{bmatrix} " + " \\\\ ".join(
            [" & ".join([f"{transition_matrix[i, j]:.3f}" for j in range(transition_matrix.shape[1])]) for i in range(transition_matrix.shape[0])]
        ) + " \\end{bmatrix}"
        st.latex(matrix_latex)

        vals, vecs = np.linalg.eig(transition_matrix)
        k = np.argmin(np.abs(vals - 1))
        pi = np.real(vecs[:, k])
        if pi.sum() < 0:
            pi = -pi
        pi = np.maximum(pi, 0)
        pi = pi / pi.sum()
        
        st.subheader("Stationary Distribution")
        pi_df = pd.DataFrame({"Node": st.session_state["nodes"], "π (steady state)": pi})
        st.dataframe(pi_df, hide_index=True)

        st.subheader("Convergence to the Stationary Distribution")
        num_steps = 50
        n = transition_matrix.shape[0]
        dist = np.ones(n) / n
        history = [dist.copy()]
        for _ in range(num_steps):
            dist = transition_matrix @ dist
            history.append(dist.copy())
        hist_arr = np.vstack(history)
        conv_df = pd.DataFrame(hist_arr, columns=st.session_state["nodes"])
        conv_df.index.name = "step"
        st.line_chart(conv_df)

        