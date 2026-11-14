import argparse
import os
from html import escape
from pathlib import Path
import webbrowser

import igraph
import numpy as np
import plotly.graph_objects as go

from rstar_deepthink.tools.python_tool import comment_out_markers, remove_markers
from utils import load_nodes


def build_graph_from_nodes(nodes, step_only=False):
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
        code = (remove_markers(node.state["code"]) if step_only
                else comment_out_markers(node.collect_code()))
        if step_only:
            code = "\n".join(line for line in code.splitlines() if line.strip())
        q_value = node.q_value()
        data = (f"Tag: {node.tag}<br>Value: {node.value}<br>Visit count: {node.visit_count}"
                f"<br>Value sum: {node.value_sum}<br>Q-Value: {q_value}<br>PUCT: {node.puct()}")
        if node.terminal_reason is not None:
            data += f"<br>Terminal reason: {escape(node.terminal_reason)}"
        if node.temperature is not None:
            data += f"<br>Temperature: {node.temperature}"
        if node.example_name is not None:
            data += f"<br>Example name: {escape(str(node.example_name))}"
        code = escape(code).replace('\n', '<br>')
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


def compute_layout(G):
    """Return layout positions and edge coordinates for the tree graph."""
    layout = G.layout("rt")
    positions = {idx: layout[idx] for idx in range(len(G.vs))}
    y_coords = [layout[idx][1] for idx in range(len(G.vs))]
    max_y = max(y_coords)

    Xn = [positions[idx][0] for idx in positions]
    Yn = [2 * max_y - positions[idx][1] for idx in positions]

    Xe, Ye = [], []
    for edge in G.es:
        start, end = edge.tuple
        Xe += [positions[start][0], positions[end][0], None]
        Ye += [2 * max_y - positions[start][1], 2 * max_y - positions[end][1], None]

    return Xn, Yn, Xe, Ye


def _add_scatter_nodes(fig, G, nodes, Xn, Yn):
    size = 22 if len(nodes) < 64 else (14 if len(nodes) < 256 else 8)
    marker = dict(symbol="circle-dot", size=size,
                  color=[G.vs[idx]["color"] for idx in range(len(G.vs))],
                  line=dict(color="rgb(50,50,50)", width=1))

    normal_node_trace = go.Scatter(
        x=Xn,
        y=Yn,
        mode="markers",
        marker=marker,
        text=[G.vs[idx]["normal_hovertext"] for idx in range(len(G.vs))],
        hoverinfo="text",
        hovertemplate="%{text}<extra></extra>",
        opacity=0.8,
        name="normal_nodes",
        visible=True,
    )

    shift_node_trace = go.Scatter(
        x=Xn,
        y=Yn,
        mode="markers",
        marker=marker,
        text=[G.vs[idx]["shift_hovertext"] for idx in range(len(G.vs))],
        hoverinfo="text",
        hovertemplate="%{text}<extra></extra>",
        opacity=0.8,
        name="shift_nodes",
        visible=False,
    )

    fig.add_trace(normal_node_trace)
    fig.add_trace(shift_node_trace)


def create_figure(G, nodes, Xn, Yn, Xe, Ye, theme="dark"):
    """Create Plotly figure for the given tree graph."""
    light = theme == "light"
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(150,150,150)" if light else "rgb(210,210,210)", width=1),
            hoverinfo="none",
            name="edges",
        )
    )

    _add_scatter_nodes(fig, G, nodes, Xn, Yn)

    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False)
    fig.update_layout(
        title="Tree Visualization (Press Shift to see alternative hover text)",
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode="closest",
        plot_bgcolor="white" if light else "rgb(30,30,30)",
        paper_bgcolor="white" if light else "rgb(30,30,30)",
        font_color="black" if light else "white",
        hoverlabel=dict(
            bgcolor="white" if light else "black",
            font_size=12,
            font_family="monospace",
            font_color="black" if light else "white",
            align="left",
            bordercolor="#cccccc" if light else "#333333",
            namelength=-1,
        ),
    )

    return fig


def visualize_tree(json_filename, open_website=False, output_html=None, theme="dark", step_only=False):
    """Visualize a serialized node tree using Plotly."""
    nodes = load_nodes(json_filename)
    if not nodes:
        print("No nodes loaded.")
        return

    graph = build_graph_from_nodes(nodes, step_only=step_only)
    Xn, Yn, Xe, Ye = compute_layout(graph)

    fig = create_figure(graph, nodes, Xn, Yn, Xe, Ye, theme=theme)

    # Add a custom JavaScript to switch trace visibility on shift key press
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var isShiftPressed = false;
        var plotlyDiv = document.getElementById('plotly-div');


        // Add event listeners for shift key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Shift' && !isShiftPressed) {
                isShiftPressed = true;

                // Hide normal nodes, show shift nodes
                Plotly.restyle(plotlyDiv, {'visible': false}, [1]);  // Normal nodes (index 1)
                Plotly.restyle(plotlyDiv, {'visible': true}, [2]);   // Shift nodes (index 2)
            }
        });

        document.addEventListener('keyup', function(e) {
            if (e.key === 'Shift') {
                isShiftPressed = false;

                // Show normal nodes, hide shift nodes
                Plotly.restyle(plotlyDiv, {'visible': true}, [1]);   // Normal nodes
                Plotly.restyle(plotlyDiv, {'visible': false}, [2]);  // Shift nodes
            }
        });

        // Safeguard: Handle cases where shift key is released outside the window
        window.addEventListener('blur', function() {
            if (isShiftPressed) {
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
    output_html = Path(output_html) if output_html else Path(json_dir) / f"tree_{nodes[0].task}_visualization.html"
    output_html.parent.mkdir(parents=True, exist_ok=True)

    # Generate the HTML file with custom div ID and our JS script
    with open(output_html, 'w') as f:
        # Get the base HTML from plotly but add our own div ID
        plot_html = fig.to_html(include_plotlyjs=True, full_html=True, div_id='plotly-div')

        # Insert our custom JavaScript before the closing body tag
        modified_html = plot_html.replace('</body>', custom_js + '</body>')

        f.write(modified_html)

    if open_website:
        webbrowser.open(output_html.resolve().as_uri())
    return output_html


def main():
    parser = argparse.ArgumentParser(description="Render saved rStar-ARC search trees as interactive HTML.")
    parser.add_argument("--input", required=True, type=Path, help="Saved node JSON file or directory of *_nodes.json files")
    parser.add_argument("--output", type=Path, help="HTML file, or output directory when rendering a directory")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    parser.add_argument("--step-only", action="store_true", help="Show each node's own code instead of the full trajectory")
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in a browser")
    args = parser.parse_args()

    if args.input.is_file():
        inputs = [args.input]
    elif args.input.is_dir():
        inputs = sorted(args.input.glob("*_nodes.json"))
        if not inputs:
            parser.error(f"No *_nodes.json files in {args.input}")
    else:
        parser.error(f"Input does not exist: {args.input}")

    for path in inputs:
        output = args.output
        if output and args.input.is_dir():
            output = output / f"{path.stem}.html"
        result = visualize_tree(path, open_website=args.open, output_html=output,
                                theme=args.theme, step_only=args.step_only)
        if result:
            print(result)


if __name__ == "__main__":
    main()
