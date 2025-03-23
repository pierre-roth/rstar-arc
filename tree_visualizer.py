import os
import igraph
import webbrowser
import plotly.graph_objects as go
from utils import load_nodes
from arc_rstar.tools.python_tool import extract_python_code


def build_graph_from_nodes(nodes):
    """
    Create an igraph.Graph from a list of nodes.
    It assumes that each node has a unique 'tag' attribute and that the
    'parent' attribute is set (None for root) so that an edge exists from the parent
    to the child.
    """
    # Create a list of tags for vertices.
    tags = [node.tag for node in nodes]
    # Create a list of hover texts from nodes using node.state["text"]
    hover_texts = []
    for node in nodes:
        code = extract_python_code(node.collect_partial_solution())
        # Replace newlines with Plotly's <br> tag for line breaks
        formatted_code = code.replace('\n', '<br>')
        hover_texts.append(formatted_code)

    # Create node colors based on some attribute (example)
    node_colors = []
    for node in nodes:
        # Example condition - replace with your own attribute check
        # For example: node.state.get("success", False)
        if node.valid and node.terminal_reason is not None and node.passes_training:
            node_colors.append('#00cc00')  # Green for successful nodes
        elif not node.valid:
            node_colors.append('#ff0000')  # Red for invalid nodes
        elif node.terminal_reason is not None:
            node_colors.append('#ffcc00')  # Yellow for terminal nodes
        else:
            node_colors.append('#6175c1')  # Default blue for other nodes

    tag_to_index = {tag: idx for idx, tag in enumerate(tags)}

    # Build edge list: add an edge from parent to node for each node with a parent.
    edges = []
    for node in nodes:
        if node.parent is not None:
            parent_idx = tag_to_index[node.parent.tag]
            child_idx = tag_to_index[node.tag]
            edges.append((parent_idx, child_idx))

    # Create a directed graph.
    G = igraph.Graph(directed=True)
    G.add_vertices(len(tags))
    G.vs["label"] = tags
    G.vs["hovertext"] = hover_texts
    G.vs["color"] = node_colors
    if edges:
        G.add_edges(edges)
    return G


def visualize_tree(json_filename):
    # Load nodes from file.
    nodes = load_nodes(json_filename)
    if not nodes:
        print("No nodes loaded.")
        return

    # Build graph from nodes.
    G = build_graph_from_nodes(nodes)

    # Compute a tree layout. The Reingold-Tilford layout is appropriate for trees.
    layout = G.layout("rt")
    positions = {idx: layout[idx] for idx in range(len(G.vs))}
    # Use the y-coordinate to get a maximum for reversing y (so that the root is on top).
    Y = [layout[idx][1] for idx in range(len(G.vs))]
    M = max(Y)

    # Build lists for node and edge coordinates.
    Xn = [positions[idx][0] for idx in positions]
    Yn = [2 * M - positions[idx][1] for idx in positions]

    # Build edge coordinates.
    Xe = []
    Ye = []
    for edge in G.es:
        start, end = edge.tuple
        Xe += [positions[start][0], positions[end][0], None]
        Ye += [2 * M - positions[start][1], 2 * M - positions[end][1], None]

    # Create annotations: label each node with its tag.
    annotations = []
    for idx in positions:
        annotations.append(dict(
            text=G.vs[idx]["label"],
            x=positions[idx][0],
            y=2 * M - positions[idx][1],
            xref='x', yref='y',
            font=dict(color='rgb(250,250,250)', size=10),
            showarrow=False
        ))

    # Create Plotly figure.
    fig = go.Figure()
    # Add edge traces.
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'))
    # Add node traces with individual colors.
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             marker=dict(symbol='circle-dot',
                                         size=18,
                                         color=[G.vs[idx]["color"] for idx in range(len(G.vs))],
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             text=G.vs["hovertext"],
                             hoverinfo='text',
                             hovertemplate='%{text}<extra></extra>',
                             opacity=0.8))

    # Layout settings.
    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False)
    # Add more comprehensive hoverlabel settings with black background and white text
    fig.update_layout(title='Tree Visualization',
                      annotations=annotations,
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)',
                      hoverlabel=dict(
                          bgcolor="black",  # Changed to black background
                          font_size=12,
                          font_family="monospace",
                          font_color="white",  # Added white font color
                          align="left",
                          bordercolor="#333333",  # Darker border to match the black background
                          namelength=-1  # Show the full text without truncating
                      ))

    # Save the HTML in the same directory as the JSON file.
    json_dir = os.path.dirname(os.path.abspath(json_filename))
    output_html = os.path.join(json_dir, f"tree_{nodes[0].task}_visualization.html")
    fig.write_html(output_html)
    webbrowser.open('file://' + os.path.realpath(output_html))


if __name__ == "__main__":
    visualize_tree(input("Enter the path to the JSON file: "))