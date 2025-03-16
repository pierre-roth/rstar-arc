import json
import plotly.graph_objects as go
import igraph as ig
import argparse
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import sys

# Add a nice banner when program starts
BANNER = """
╔═══════════════════════════════════════════════════════╗
║                  MCTS Tree Visualizer                 ║
║                                                       ║
║  Visualize Monte Carlo Tree Search trees from logs    ║
╚═══════════════════════════════════════════════════════╝
"""


def parse_log_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse the log file and extract node information.

    Specifically looks for lines containing JSON output from Node add_child method.
    Each node should have a tag (e.g., "0", "0.1", "0.1.2") and data attributes like
    visits, value, reward, depth, prior_probability, etc.

    Args:
        filepath: Path to the log file.

    Returns:
        List of node dictionaries.
    """
    """
    Parse the log file and extract node information.

    Args:
        filepath: Path to the log file.

    Returns:
        List of node dictionaries.
    """
    nodes = []

    with open(filepath, 'r') as file:
        for line in file:
            # Try to find a JSON object in each line
            try:
                # Find opening and closing braces
                start_idx = line.find('{')
                end_idx = line.rfind('}')

                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    # Extract potential JSON string
                    json_str = line[start_idx:end_idx + 1]
                    node_data = json.loads(json_str)

                    # Check if it has the expected node format
                    if "node" in node_data and isinstance(node_data["node"], str):
                        # Add to our list of nodes
                        nodes.append(node_data)
            except (json.JSONDecodeError, ValueError):
                # Not a valid JSON, skip
                continue

    return nodes


def build_tree(nodes: List[Dict[str, Any]], max_depth: Optional[int] = None) -> Tuple[ig.Graph, Dict[str, int]]:
    """
    Build an igraph tree from the list of nodes.

    Args:
        nodes: List of node dictionaries.
        max_depth: Maximum depth of nodes to include (optional).

    Returns:
        Tuple of (igraph Graph object, node name to index mapping).
    """
    # Create a dictionary of node name to data
    node_data_map = {node["node"]: node.get("data", {}) for node in nodes}

    # Filter by depth if specified
    if max_depth is not None:
        node_data_map = {name: data for name, data in node_data_map.items()
                         if name.count('.') < max_depth}

    # Create a mapping of node names to indices
    node_indices = {}

    # Create graph
    g = ig.Graph(directed=True)

    # First, add the root node if it exists
    if "0" in node_data_map:
        g.add_vertex(name="0", **node_data_map["0"])
        node_indices["0"] = 0
    else:
        # Create a default root node
        g.add_vertex(name="0")
        node_indices["0"] = 0

    # Add remaining nodes and edges
    for node_name, data in sorted(node_data_map.items(), key=lambda x: len(x[0].split('.'))):
        if node_name == "0":
            continue  # Skip root as it's already added

        # Add the node
        node_indices[node_name] = len(g.vs)
        g.add_vertex(name=node_name, **data)

        # Find and connect to parent
        parts = node_name.split('.')
        if len(parts) > 1:
            parent_name = '.'.join(parts[:-1])
            # Ensure parent exists
            if parent_name not in node_indices:
                # Check if parent is above max_depth
                if max_depth is not None and parent_name.count('.') >= max_depth:
                    continue

                # Create parent node with default data
                node_indices[parent_name] = len(g.vs)
                g.add_vertex(name=parent_name)

            # Connect to parent
            parent_idx = node_indices[parent_name]
            g.add_edge(parent_idx, node_indices[node_name])

    return g, node_indices


def visualize_tree(g: ig.Graph, color_attr: Optional[str] = 'value',
                   node_size_attr: Optional[str] = 'visits', output_file: Optional[str] = None):
    """
    Visualize the tree using Plotly and igraph.

    Args:
        g: igraph Graph object.
        color_attr: Attribute to use for coloring nodes (optional).
        node_size_attr: Attribute to use for node sizes (optional).
        output_file: Path to save the visualization (optional).
    """
    # Calculate layout using igraph's tree layout
    layout = g.layout_reingold_tilford(root=[0])
    layout = np.array(layout.coords)

    # Convert layout to x, y coordinates
    Xn, Yn = layout[:, 0], layout[:, 1]

    # Create edges
    Xe, Ye = [], []
    for edge in g.es:
        source, target = edge.tuple
        Xe += [Xn[source], Xn[target], None]
        Ye += [Yn[source], Yn[target], None]

    # Create node labels and hover text
    labels = g.vs["name"]
    hover_texts = []

    for v in g.vs:
        hover_text = f"Node: {v['name']}<br>"

        # Create organized sections for hover text
        core_attrs = []
        mcts_attrs = []
        validation_attrs = []
        other_attrs = []

        for attr, value in v.attributes().items():
            if attr in ["name"]:
                continue

            # Format the value based on type
            if isinstance(value, float):
                display_value = f"{value:.4f}"
            else:
                display_value = value

            # Group attributes by category
            if attr in ["depth", "has_children"]:
                core_attrs.append(f"{attr}: {display_value}")
            elif attr in ["visits", "value", "reward", "prior_probability", "puct_score"]:
                mcts_attrs.append(f"{attr}: {display_value}")
            elif attr in ["is_valid", "terminal_reason"]:
                validation_attrs.append(f"{attr}: {display_value}")
            else:
                other_attrs.append(f"{attr}: {display_value}")

        # Add sections to hover text
        if core_attrs:
            hover_text += "<br><b>Core:</b><br>" + "<br>".join(core_attrs)
        if mcts_attrs:
            hover_text += "<br><b>MCTS:</b><br>" + "<br>".join(mcts_attrs)
        if validation_attrs:
            hover_text += "<br><b>Validation:</b><br>" + "<br>".join(validation_attrs)
        if other_attrs:
            hover_text += "<br><b>Other:</b><br>" + "<br>".join(other_attrs)

        hover_texts.append(hover_text)

    # Calculate node colors if color_attr is provided
    visual_info = []
    if color_attr and color_attr in g.vs.attributes():
        print(f"Using '{color_attr}' for node coloring")
        # Get attribute values
        attr_values = g.vs[color_attr]

        # Check if values are numeric
        if all(isinstance(val, (int, float)) for val in attr_values if val is not None):
            # Replace None with 0 for calculations
            safe_values = [val if val is not None else 0 for val in attr_values]

            # Normalize values between 0 and 1
            min_val = min(safe_values)
            max_val = max(safe_values)

            if min_val == max_val:
                normalized = [0.5 for _ in safe_values]
            else:
                normalized = [(val - min_val) / (max_val - min_val) for val in safe_values]

            # Create a blue color scale
            colors = [f'rgb({int(255 * (1 - n))}, {int(255 * (1 - n))}, 255)' for n in normalized]
        else:
            # For non-numeric values, use a categorical colormap
            # Filter out None values first
            non_none_values = [val for val in attr_values if val is not None]

            if not non_none_values:
                # All values are None, use default color
                colors = ['skyblue'] * len(g.vs)
            else:
                unique_values = list(set(non_none_values))
                color_map = {val: i / max(1, len(unique_values) - 1) for i, val in enumerate(unique_values)}
                normalized = [color_map.get(val, 0) if val is not None else 0 for val in attr_values]

                # Create colors using a hue range - green to blue
                colors = [f'hsl({int(120 + 120 * n)}, 80%, 70%)' for n in normalized]

        visual_info.append(f"Color: '{color_attr}'")
    else:
        # Default blue color
        colors = ['skyblue'] * len(g.vs)

    # Calculate node sizes
    if node_size_attr and node_size_attr in g.vs.attributes():
        print(f"Using '{node_size_attr}' for node sizing")
        # Get attribute values
        attr_values = g.vs[node_size_attr]

        # Check if values are numeric
        if all(isinstance(val, (int, float)) for val in attr_values if val is not None):
            # Replace None with 0 for calculations
            safe_values = [val if val is not None else 0 for val in attr_values]

            # Normalize values between min and max size
            min_val = min(safe_values)
            max_val = max(safe_values)

            min_size, max_size = 10, 30

            if min_val == max_val:
                node_sizes = [20] * len(g.vs)
            else:
                node_sizes = [
                    min_size + (max_size - min_size) * (val - min_val) / (max_val - min_val)
                    for val in safe_values
                ]
        else:
            # Default size
            node_sizes = [15] * len(g.vs)

        visual_info.append(f"Size: '{node_size_attr}'")
    else:
        # Safely get depth from node name
        node_sizes = []
        for name in labels:
            try:
                depth = name.count('.')
                size = max(10, min(25, 20 - 3 * depth))
            except (AttributeError, TypeError):
                size = 15  # Default size if name isn't a string or has issues
            node_sizes.append(size)

    # Only show labels for nodes up to depth 2 if many nodes
    if len(g.vs) > 50:
        # For larger trees, only show labels for upper levels
        node_texts = []
        for name in labels:
            try:
                depth = name.count('.')
                text = name if depth < 3 else ""
            except (AttributeError, TypeError):
                text = ""  # Default empty if name isn't a string
            node_texts.append(text)
    else:
        node_texts = labels

    # Augment the graph with any calculated values
    # Ensure all numeric attributes have default values instead of None
    numeric_attrs = ['visits', 'value', 'reward', 'prior_probability', 'depth']
    for attr in numeric_attrs:
        if attr in g.vs.attributes():
            for i, v in enumerate(g.vs):
                if v[attr] is None:
                    g.vs[i][attr] = 0.0 if attr != 'depth' else 0

    # Calculate PUCT scores if not already present but components exist
    if 'value' in g.vs.attributes() and 'prior_probability' in g.vs.attributes() and 'visits' in g.vs.attributes():
        if 'puct_score' not in g.vs.attributes():
            g.vs['puct_score'] = [0.0] * len(g.vs)
            for i, v in enumerate(g.vs):
                # Simple approximation of PUCT score for visualization
                visits = v['visits'] if v['visits'] is not None else 0
                if visits > 0 and i > 0:  # Skip root node
                    neighbors = g.neighbors(i, mode='in')
                    parent_visits = g.vs[neighbors[0]]['visits'] if neighbors else 0
                    parent_visits = parent_visits if parent_visits is not None else 0

                    if parent_visits > 0:
                        prior = v['prior_probability'] if v['prior_probability'] is not None else 1.0
                        value = v['value'] if v['value'] is not None else 0.0
                        exploration = 1.0 * prior * math.sqrt(math.log(parent_visits)) / (1 + visits)
                        v['puct_score'] = value + exploration

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=Xe,
        y=Ye,
        mode='lines',
        line=dict(color='rgba(0,0,0,0.5)', width=1),
        hoverinfo='none'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=colors,
            line=dict(color='black', width=1)
        ),
        text=node_texts,
        hovertext=hover_texts,
        hoverinfo='text',
        textposition='top center'
    ))

    # Update layout
    title = 'MCTS Tree Visualization'
    if visual_info:
        title += f' ({", ".join(visual_info)})'

    fig.update_layout(
        title=title,
        plot_bgcolor='white',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Show or save figure
    if output_file:
        fig.write_html(output_file)
    else:
        fig.show()


def generate_sample_json():
    """
    Generate a sample JSON format to use for node information.

    Returns:
        A string containing a sample JSON node.
    """
    sample = {
        "node": "0.1.2",
        "data": {
            # Core MCTS attributes
            "visits": 150,
            "value": 0.75,
            "reward": 0.85,
            "depth": 3,

            # Node class specific attributes
            "prior_probability": 1.0,
            "is_valid": True,
            "terminal_reason": None,
            "has_children": True,

            # Additional attributes that might be useful
            "puct_score": 1.25  # Could be calculated using the puct_score() method
        }
    }
    return json.dumps(sample, indent=2)


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description='Visualize MCTS tree from log file.')
    parser.add_argument('log_file', nargs='?', help='Path to the log file.')
    parser.add_argument('--output', '-o', help='Path to save the visualization (optional).')
    parser.add_argument('--color', '-c', help='Node attribute to use for coloring (optional).')
    parser.add_argument('--size', '-s', help='Node attribute to use for sizing (optional).')
    parser.add_argument('--max-depth', '-d', type=int, help='Maximum depth of nodes to include (optional).')
    parser.add_argument('--sample', action='store_true', help='Print a sample JSON format for node data.')

    args = parser.parse_args()

    if args.sample:
        print("Sample JSON format for node data:")
        print(generate_sample_json())
        print("\nEach line in the log file should contain a JSON object with this format.")
        print("Other lines will be ignored.")
        return

    if not args.log_file:
        parser.print_help()
        return

    # Parse log file
    try:
        print(f"Parsing log file: {args.log_file}")
        nodes = parse_log_file(args.log_file)
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        return

    if not nodes:
        print("No valid node data found in the log file.", file=sys.stderr)
        return

    print(f"Found {len(nodes)} nodes in the log file.")

    # Build tree
    try:
        print("Building tree...")
        graph, node_indices = build_tree(nodes, args.max_depth)
        print(f"Tree built with {len(graph.vs)} nodes and {len(graph.es)} edges.")
    except Exception as e:
        print(f"Error building tree: {e}", file=sys.stderr)
        return

    # Visualize tree
    try:
        print("Generating visualization...")
        visualize_tree(graph, args.color, args.size, args.output)

        if args.output:
            print(f"Visualization saved to: {args.output}")
        else:
            print("Displaying visualization...")
    except Exception as e:
        print(f"Error visualizing tree: {e}", file=sys.stderr)
        return


if __name__ == "__main__":
    main()

