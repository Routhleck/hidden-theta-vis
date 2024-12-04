from functools import partial

import networkx as nx
import holoviews as hv
from holoviews import opts
from bokeh.io import show

from data import Hidden, Theta, generate_data

hv.extension('bokeh')

# RENDER SETTINGS
HIDDEN_NODE_SIZE = 18
THETA_NODE_SIZE = 12
HIDDEN_NODE_COLOR = '#4daf4a'  # green
THETA_NODE_COLOR = '#e41a1c'  # red
HIDDEN_EDGE_COLOR = '#377eb8'  # blue
THETA_HIDDEN_EDGE_COLOR = '#ff7f00'  # orange
THETA_THETA_EDGE_COLOR = '#984ea3'  # purple
HIDDEN_EDGE_WEIGHT = 2.5
THETA_EDGE_WEIGHT = 2



# Create a directed graph from the generated data
def create_graph(hiddens: list[Hidden], thetas: list[Theta]) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add Hidden nodes with larger size and specific color
    for hidden in hiddens:
        G.add_node(f"Hidden_{hidden.id}", type="Hidden", size=HIDDEN_NODE_SIZE, color=HIDDEN_NODE_COLOR)

    # Add Theta nodes with smaller size and specific color
    for theta in thetas:
        G.add_node(f"Theta_{theta.id}", type="Theta", size=THETA_NODE_SIZE, color=THETA_NODE_COLOR)

    # Add edges for Hidden targets with specific color
    for hidden in hiddens:
        if hidden.target:
            G.add_edge(f"Hidden_{hidden.id}", f"Hidden_{hidden.target.id}", color=HIDDEN_EDGE_COLOR,
                       weight=HIDDEN_EDGE_WEIGHT)

    # Add edges for Theta targets with specific color
    for theta in thetas:
        if theta.target:
            if isinstance(theta.target, Hidden):
                G.add_edge(f"Theta_{theta.id}", f"Hidden_{theta.target.id}",
                           color=THETA_HIDDEN_EDGE_COLOR, weight=THETA_EDGE_WEIGHT)
            elif isinstance(theta.target, Theta):
                G.add_edge(f"Theta_{theta.id}", f"Theta_{theta.target.id}",
                           color=THETA_THETA_EDGE_COLOR, weight=THETA_EDGE_WEIGHT)

    return G

# Visualize the graph
def visualize_graph(G: nx.DiGraph, file_name: str='graph_spring_layout.html'):
    custom_layout = partial(nx.layout.spring_layout, k=1, iterations=250, seed=42, threshold=1e-4)  # Custom layout for the graph
    graph = hv.Graph.from_networkx(G, custom_layout).opts(
        opts.Graph(directed=True, node_size='size', bgcolor='#1e1e1e', xaxis=None, yaxis=None,
                   edge_line_color='color', edge_line_width=1, width=800, height=800, arrowhead_length=0.01,
                   node_fill_color='color', node_nonselection_fill_color='#7f7f7f', node_selection_fill_color='#d62728')
    )

    hv.save(graph, file_name, fmt='html')
    hv.render(graph)

if __name__ == '__main__':
    # Example usage
    num_hiddens = 25
    num_hidden_groups = 6
    num_thetas = 50

    hiddens, hidden_groups, thetas = generate_data(num_hiddens, num_hidden_groups, num_thetas)

    # Create and visualize the graph
    G = create_graph(hiddens, thetas)
    file_name = visualize_graph(G)