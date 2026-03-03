# Generate LaTeX code for the graph
def generate_latex_graph(nodes, edges):
    tikz_nodes = [f"\\node[draw, circle] ({n}) at ({i*2},0) {{$ {n} $}};" for i, n in enumerate(nodes)]
    tikz_edges = [f"\\draw[->] ({start}) -- node[above] {{$ {cost} $}} ({end});" for start, end, cost in edges]
    tikz_code = "\\begin{tikzpicture}[>=stealth]\n" + "\n".join(tikz_nodes + tikz_edges) + "\n\\end{tikzpicture}"
    return tikz_code

st.subheader("Graph Visualization (LaTeX/TikZ)")
latex_code = generate_latex_graph(list(st.session_state.nodes), st.session_state.edges)
st.latex(latex_code)