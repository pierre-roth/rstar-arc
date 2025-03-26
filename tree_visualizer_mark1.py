import os
import webbrowser

import igraph
import numpy as np
import plotly.graph_objects as go

from arc_rstar.tools.python_tool import extract_python_code
from config import STEP_END, CODE_END
from utils import load_nodes


def build_graph_from_nodes(nodes):
    """
    Create an igraph.Graph from a list of nodes.
    It assumes that each node has a unique 'tag' attribute and that the
    'parent' attribute is set (None for root) so that an edge exists from the parent
    to the child.
    """
    # Create a list of tags for vertices.
    tags = [node.tag for node in nodes]

    # Create two separate lists for different hover texts
    normal_hover_texts = []
    shift_hover_texts = []

    for node in nodes:
        # Extract raw Python code and other data
        try:
            code = extract_python_code(node.collect_partial_solution())
            q_value = node.q_value() if node.visit_count > 0 else 0

            # Handle the case when parent is None (root node)
            if not node.parent:
                puct = q_value  # For root node, PUCT is just the Q-value
            else:
                # Calculate PUCT only when parent exists
                if node.parent.visit_count == 0 or node.visit_count == 0:
                    u_value = 0
                else:
                    u_value = node.config["c_puct"] * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
                puct = q_value + u_value

            data = f"Tag: {node.tag}<br>Value: {node.value}<br>Visit count: {node.visit_count}<br>Value sum: {node.value_sum}<br>Q-Value: {q_value}<br>PUCT: {puct}"
        except Exception as e:
            print(f"Error while extracting code for node {node.tag}: {e}")
            code = "NO VALID CODE FOUND!"
            data = "NO DATA FOUND!"
        # Replace newlines with <br> tags for multi‑line hover text.
        code = code.replace('\n', '<br>').replace(STEP_END, f"# {STEP_END}<br>").replace(CODE_END, "")
        normal_hover_texts.append(code)
        shift_hover_texts.append(data)

    # Create node colors based on some attribute (example)
    node_colors = []
    for node in nodes:
        if node.is_valid_final_answer_node():
            node_colors.append('#00cc00')  # Green for successful nodes
        elif not node.valid:
            node_colors.append('#ff0000')  # Red for invalid nodes
        elif node.terminal:
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
    G.add_vertices(n=len(tags))
    G.vs["label"] = tags
    G.vs["normal_hovertext"] = normal_hover_texts  # Store normal hover text
    G.vs["shift_hovertext"] = shift_hover_texts  # Store shift hover text
    G.vs["color"] = node_colors
    if edges:
        G.add_edges(es=edges)
    return G


def visualize_tree(json_filename, open_website=True):
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
    Y = [layout[idx][1] for idx in range(len(G.vs))]
    M = max(Y)

    Xn = [positions[idx][0] for idx in positions]
    Yn = [2 * M - positions[idx][1] for idx in positions]

    Xe = []
    Ye = []
    for edge in G.es:
        start, end = edge.tuple
        Xe += [positions[start][0], positions[end][0], None]
        Ye += [2 * M - positions[start][1], 2 * M - positions[end][1], None]

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=Xe,
        y=Ye,
        mode='lines',
        line=dict(color='rgb(210,210,210)', width=1),
        hoverinfo='none',
        name='edges'
    ))

    # Create two node traces - one for normal hover and one for shift hover
    normal_node_trace = go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers',
        marker=dict(
            symbol='circle-dot',
            size=18 if len(nodes) < 64 else (12 if len(nodes) < 256 else 6),
            color=[G.vs[idx]["color"] for idx in range(len(G.vs))],
            line=dict(color='rgb(50,50,50)', width=1)
        ),
        text=[G.vs[idx]["normal_hovertext"] for idx in range(len(G.vs))],
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        opacity=0.8,
        name='normal_nodes',
        visible=True
    )

    shift_node_trace = go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers',
        marker=dict(
            symbol='circle-dot',
            size=18 if len(nodes) < 64 else (12 if len(nodes) < 256 else 6),
            color=[G.vs[idx]["color"] for idx in range(len(G.vs))],
            line=dict(color='rgb(50,50,50)', width=1)
        ),
        text=[G.vs[idx]["shift_hovertext"] for idx in range(len(G.vs))],
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',
        opacity=0.8,
        name='shift_nodes',
        visible=False  # Initially hidden
    )

    fig.add_trace(normal_node_trace)
    fig.add_trace(shift_node_trace)

    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False)
    fig.update_layout(
        title='Tree Visualization (Press Shift to see alternative hover text)',
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode='closest',
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(30,30,30)',
        font_color='white',
        hoverlabel=dict(
            bgcolor="black",
            font_size=12,
            font_family="monospace",
            font_color="white",
            align="left",
            bordercolor="#333333",
            namelength=-1
        )
    )

    # Add a custom JavaScript to switch trace visibility on shift key press
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var isShiftPressed = false;
        var plotlyDiv = document.getElementById('plotly-div');

        console.log("Shift key handler initialized");

        // Add event listeners for shift key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Shift' && !isShiftPressed) {
                console.log("Shift key pressed down");
                isShiftPressed = true;

                // Hide normal nodes, show shift nodes
                Plotly.restyle(plotlyDiv, {'visible': false}, [1]);  // Normal nodes (index 1)
                Plotly.restyle(plotlyDiv, {'visible': true}, [2]);   // Shift nodes (index 2)
            }
        });

        document.addEventListener('keyup', function(e) {
            if (e.key === 'Shift') {
                console.log("Shift key released");
                isShiftPressed = false;

                // Show normal nodes, hide shift nodes
                Plotly.restyle(plotlyDiv, {'visible': true}, [1]);   // Normal nodes
                Plotly.restyle(plotlyDiv, {'visible': false}, [2]);  // Shift nodes
            }
        });

        // Safeguard: Handle cases where shift key is released outside the window
        window.addEventListener('blur', function() {
            if (isShiftPressed) {
                console.log("Window blur - resetting shift state");
                isShiftPressed = false;

                // Show normal nodes, hide shift nodes
                Plotly.restyle(plotlyDiv, {'visible': true}, [1]);   // Normal nodes
                Plotly.restyle(plotlyDiv, {'visible': false}, [2]);  // Shift nodes
            }
        });
    });
    </script>
    """

    json_dir = os.path.dirname(os.path.abspath(json_filename))
    output_html = os.path.join(json_dir, f"tree_{nodes[0].task}_visualization.html")

    # Generate the HTML file with custom div ID and our JS script
    with open(output_html, 'w') as f:
        # Get the base HTML from plotly but add our own div ID
        plot_html = fig.to_html(include_plotlyjs=True, full_html=True, div_id='plotly-div')

        # Insert our custom JavaScript before the closing body tag
        modified_html = plot_html.replace('</body>', custom_js + '</body>')

        f.write(modified_html)

    if open_website:
        webbrowser.open('file://' + os.path.realpath(output_html))


if __name__ == "__main__":
    input_path = input("Enter the path to the JSON file: ")

    if os.path.isfile(input_path):
        visualize_tree(input_path)
    else:
        open_all = input("Do you want to open all visualizations? (y/n): ")
        for filename in os.listdir(input_path):
            if filename.endswith(".json"):
                visualize_tree(os.path.join(input_path, filename), open_website=open_all.strip().lower() == "y")
