import json
import plotly.graph_objects as go
import igraph as ig
import argparse
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import sys
import traceback

# Import terminal constants from config
from config import TERMINAL_CODE_END, TERMINAL_MAX_DEPTH, TERMINAL_INVALID, TERMINAL_FAILURE, TERMINAL_SUCCESS

# Add a nice banner when program starts
BANNER = """
╔═══════════════════════════════════════════════════════╗
║                Dark Mode Tree Visualizer              ║
║                                                       ║
║         Visualize tree structures from logs           ║
╚═══════════════════════════════════════════════════════╝
"""

# Define terminal reason colors for dark mode
TERMINAL_COLORS = {
    TERMINAL_CODE_END: '#00e676',  # bright green
    TERMINAL_MAX_DEPTH: '#ffab40',  # bright orange
    TERMINAL_INVALID: '#ff5252',  # bright red
    TERMINAL_FAILURE: '#ea80fc',  # bright purple
    TERMINAL_SUCCESS: '#40c4ff',  # bright blue
    None: '#64b5f6'  # bright light blue
}


def parse_log_file(filepath: str) -> List[Dict[str, Any]]:
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
                         if name.count('.') <= max_depth}

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
                if max_depth is not None and parent_name.count('.') > max_depth:
                    continue

                # Create parent node with default data
                node_indices[parent_name] = len(g.vs)
                g.add_vertex(name=parent_name)

            # Connect to parent
            parent_idx = node_indices[parent_name]
            g.add_edge(parent_idx, node_indices[node_name])

    return g, node_indices


def calculate_node_colors(g: ig.Graph, color_attr: str = 'terminal_reason') -> List[str]:
    """
    Calculate colors for nodes based on a specified attribute.

    Args:
        g: igraph Graph object
        color_attr: Attribute to use for coloring nodes

    Returns:
        List of color values as strings
    """
    if color_attr not in g.vs.attributes():
        return ['#64b5f6'] * len(g.vs)  # Default dark mode blue

    # Get attribute values
    attr_values = g.vs[color_attr]

    # Check if values are numeric
    if all(isinstance(val, (int, float)) for val in attr_values if val is not None):
        # Numeric values - use a gradient
        safe_values = [val if val is not None else 0 for val in attr_values]
        min_val = min(safe_values)
        max_val = max(safe_values)

        if min_val == max_val:
            normalized = [0.5 for _ in safe_values]
        else:
            normalized = [(val - min_val) / (max_val - min_val) for val in safe_values]

        # Blue intensity gradient for dark mode
        colors = [f'rgb(0, {int(100 + 155 * n)}, 255)' for n in normalized]
    else:
        # Categorical values
        non_none_values = [val for val in attr_values if val is not None]

        if not non_none_values:
            return ['#64b5f6'] * len(g.vs)  # Default dark mode blue

        # Special handling for terminal_reason
        if color_attr == 'terminal_reason':
            colors = ['#64b5f6'] * len(g.vs)  # Default dark mode blue

            for i, value in enumerate(attr_values):
                if value is not None:
                    colors[i] = TERMINAL_COLORS.get(value, '#ea80fc')  # Default to purple

            print(f"Coloring nodes by terminal reason: " +
                  f"{', '.join([f'{k}: {v}' for k, v in TERMINAL_COLORS.items() if k is not None])}")
        else:
            # Generic categorical coloring for dark mode
            unique_values = list(set(non_none_values))
            color_map = {val: i / max(1, len(unique_values) - 1) for i, val in enumerate(unique_values)}
            normalized = [color_map.get(val, 0) if val is not None else 0 for val in attr_values]
            colors = [f'hsl({int(120 + 240 * n)}, 80%, 70%)' for n in normalized]

    return colors


def calculate_node_sizes(g: ig.Graph, node_size_attr: Optional[str] = 'visits') -> List[float]:
    """
    Calculate sizes for nodes based on a specified attribute.

    Args:
        g: igraph Graph object
        node_size_attr: Attribute to use for sizing nodes

    Returns:
        List of node sizes
    """
    if node_size_attr and node_size_attr in g.vs.attributes():
        attr_values = g.vs[node_size_attr]

        if all(isinstance(val, (int, float)) for val in attr_values if val is not None):
            # Replace None with 0
            safe_values = [val if val is not None else 0 for val in attr_values]

            # Normalize between min and max size
            min_val = min(safe_values)
            max_val = max(safe_values)
            min_size, max_size = 10, 30

            if min_val == max_val:
                return [20] * len(g.vs)
            else:
                return [
                    min_size + (max_size - min_size) * (val - min_val) / (max_val - min_val)
                    for val in safe_values
                ]
        else:
            return [15] * len(g.vs)
    else:
        # Size based on depth
        node_sizes = []
        for name in g.vs["name"]:
            try:
                depth = name.count('.')
                size = max(10, min(25, 20 - 3 * depth))
            except (AttributeError, TypeError):
                size = 15
            node_sizes.append(size)
        return node_sizes


def create_hover_texts(g: ig.Graph) -> List[str]:
    """
    Create hover texts for nodes.

    Args:
        g: igraph Graph object

    Returns:
        List of hover text strings
    """
    hover_texts = []

    for v in g.vs:
        hover_text = f"Node: {v['name']}<br>"

        # Organize attributes by category
        core_attrs = []
        mcts_attrs = []
        state_attrs = []
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

            # Special handling for state_text
            if attr == "state_text" and value is not None:
                state_preview = value
                if len(state_preview) > 200:
                    state_preview = state_preview[:197] + "..."
                state_attrs.append(f"<b>State Text:</b><br><pre>{state_preview}</pre>")
                continue

            # Special handling for state_extra_info
            if attr == "state_extra_info" and value:
                state_attrs.append(f"<b>Extra Info:</b><br>{value}")
                continue

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
        if state_attrs:
            hover_text += "<br>" + "<br>".join(state_attrs)
        if other_attrs:
            hover_text += "<br><b>Other:</b><br>" + "<br>".join(other_attrs)

        hover_texts.append(hover_text)

    return hover_texts


def prepare_graph_for_visualization(g: ig.Graph) -> ig.Graph:
    """
    Prepare graph for visualization by calculating derived attributes.

    Args:
        g: igraph Graph object

    Returns:
        Updated graph object
    """
    # Ensure all numeric attributes have default values
    numeric_attrs = ['visits', 'value', 'reward', 'prior_probability', 'depth']
    for attr in numeric_attrs:
        if attr in g.vs.attributes():
            for i, v in enumerate(g.vs):
                if v[attr] is None:
                    g.vs[i][attr] = 0.0 if attr != 'depth' else 0

    # Calculate PUCT scores if components exist
    if ('value' in g.vs.attributes() and
            'prior_probability' in g.vs.attributes() and
            'visits' in g.vs.attributes()):
        if 'puct_score' not in g.vs.attributes():
            g.vs['puct_score'] = [0.0] * len(g.vs)
            for i, v in enumerate(g.vs):
                visits = v['visits'] if v['visits'] is not None else 0
                if visits > 0 and i > 0:  # Skip root node
                    neighbors = g.neighbors(i, mode='in')
                    if neighbors:
                        parent_idx = neighbors[0]
                        if parent_idx < len(g.vs):
                            parent_visits = g.vs[parent_idx]['visits'] if 'visits' in g.vs[
                                parent_idx].attributes() else 0
                            parent_visits = parent_visits if parent_visits is not None else 0

                            if parent_visits > 0:
                                prior = v['prior_probability'] if v['prior_probability'] is not None else 1.0
                                value = v['value'] if v['value'] is not None else 0.0
                                exploration = 1.0 * prior * math.sqrt(math.log(parent_visits)) / (1 + visits)
                                v['puct_score'] = value + exploration

    return g


def get_node_text_display(g: ig.Graph, max_nodes: int = 50) -> List[str]:
    """
    Determine which nodes should display labels based on tree size.

    Args:
        g: igraph Graph object
        max_nodes: Maximum number of nodes to show all labels

    Returns:
        List of text labels for nodes
    """
    labels = g.vs["name"]

    if len(g.vs) > max_nodes:
        # For larger trees, only show labels for nodes at lower depths
        node_texts = []
        for name in labels:
            try:
                depth = name.count('.')
                text = name if depth < 5 else ""
            except (AttributeError, TypeError):
                text = ""
            node_texts.append(text)
        return node_texts
    else:
        return labels


def generate_javascript_for_dark_mode() -> str:
    """
    Generate JavaScript code for applying dark mode to the visualization.

    Returns:
        JavaScript code as a string
    """
    return """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Apply dark mode to the entire document
        document.body.style.backgroundColor = '#121212';
        document.body.style.color = '#e0e0e0';

        // Apply dark mode to any tooltips that appear
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.classList && node.classList.contains('plotly-tooltip')) {
                            node.style.backgroundColor = '#2d2d2d';
                            node.style.color = '#e0e0e0';
                            node.style.border = '1px solid #444444';
                        }
                    });
                }
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });
    });
    </script>
    """


def visualize_tree(g: ig.Graph, layout: str = 'tree', color_attr: str = 'terminal_reason',
                   node_size_attr: str = 'visits', output_file: Optional[str] = None):
    """
    Visualize the tree using Plotly and igraph.

    Args:
        g: igraph Graph object.
        layout: Layout algorithm ('tree', 'force', or 'circle')
        color_attr: Attribute to use for coloring nodes
        node_size_attr: Attribute to use for node sizes
        output_file: Path to save the visualization
    """
    # Prepare graph for visualization (calculate derived values)
    g = prepare_graph_for_visualization(g)

    # Calculate layout based on specified algorithm
    if layout == 'force':
        layout_coords = g.layout_fruchterman_reingold(weights=None)
        layout_name = "force-directed"
    elif layout == 'circle':
        layout_coords = g.layout_circle()
        layout_name = "circular"
    else:
        layout_coords = g.layout_reingold_tilford(root=[0])
        layout_name = "tree"

    # Convert layout to coordinates
    layout_array = np.array(layout_coords.coords)
    Xn, Yn = layout_array[:, 0], layout_array[:, 1]

    # Create edges
    Xe, Ye = [], []
    for edge in g.es:
        source, target = edge.tuple
        Xe += [Xn[source], Xn[target], None]
        Ye += [Yn[source], Yn[target], None]

    # Calculate node visual properties
    colors = calculate_node_colors(g, color_attr)
    node_sizes = calculate_node_sizes(g, node_size_attr)
    hover_texts = create_hover_texts(g)
    node_texts = get_node_text_display(g)

    # Create figure with dark mode
    fig = go.Figure()

    # Add edges - brighter for dark mode
    fig.add_trace(go.Scatter(
        x=Xe,
        y=Ye,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=1),  # Lighter for dark mode
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
            line=dict(color='#444444', width=1)  # Darker border for dark mode
        ),
        text=node_texts,
        textfont=dict(
            color='#e0e0e0'  # Light text for dark mode
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        textposition='top center'
    ))

    # Build title with visualization info
    visual_info = []
    if color_attr:
        visual_info.append(f"Color: '{color_attr}'")
    if node_size_attr:
        visual_info.append(f"Size: '{node_size_attr}'")

    title = f'Tree Visualization ({layout_name} layout)'
    if visual_info:
        title += f' ({", ".join(visual_info)})'

    # Update layout for dark mode
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#e0e0e0')  # Light title font for dark mode
        ),
        plot_bgcolor='#121212',  # Dark background
        paper_bgcolor='#121212',  # Dark background
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            gridcolor='#444444'  # Darker grid color if shown
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            gridcolor='#444444'  # Darker grid color if shown
        )
    )

    # Show or save figure
    if output_file:
        html_content = fig.to_html(include_plotlyjs='cdn')
        js_code = generate_javascript_for_dark_mode()

        # Add dark mode CSS to the HTML
        dark_mode_css = """
        <style>
            body {
                background-color: #121212;
                color: #e0e0e0;
                margin: 0;
                padding: 0;
            }
            .plotly-graph-div {
                background-color: #121212 !important;
            }
            .modebar {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
            }
            .modebar-btn path {
                fill: #e0e0e0 !important;
            }
            .modebar-btn:hover {
                background-color: #444444 !important;
            }
            .js-plotly-plot .plotly .modebar-container {
                background-color: #2d2d2d !important;
            }
        </style>
        """

        # Insert dark mode CSS and JavaScript
        html_content = html_content.replace('<head>', '<head>' + dark_mode_css)
        html_content = html_content.replace('</body>', js_code + '</body>')

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Dark mode visualization saved to: {output_file}")
    else:
        fig.show()


def generate_sample_json():
    """Generate a sample JSON format to use for node information."""
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

            # State information
            "state_text": "def solve(input_grid):\n    # Process the grid",
            "state_extra_info": "Additional information about this state",

            # Additional attributes that might be useful
            "puct_score": 1.25
        }
    }
    return json.dumps(sample, indent=2)


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description='Visualize tree structures from log file in dark mode.')
    parser.add_argument('log_file', nargs='?', help='Path to the log file.')
    parser.add_argument('--output', '-o', help='Path to save the visualization (optional).')
    parser.add_argument('--color', '-c', default='terminal_reason',
                        help='Node attribute to use for coloring (default: terminal_reason).')
    parser.add_argument('--size', '-s', default='visits',
                        help='Node attribute to use for sizing (default: visits).')
    parser.add_argument('--max-depth', '-d', type=int,
                        help='Maximum depth of nodes to include (optional, default: no limit).')
    parser.add_argument('--sample', action='store_true',
                        help='Print a sample JSON format for node data.')
    parser.add_argument('--layout', '-l', choices=['tree', 'force', 'circle'], default='tree',
                        help='Layout algorithm to use (default: tree)')

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
        print(f"Using {args.layout} layout with dark mode theme...")
        visualize_tree(
            graph,
            layout=args.layout,
            color_attr=args.color,
            node_size_attr=args.size,
            output_file=args.output
        )
    except Exception as e:
        print(f"Error visualizing tree: {e}", file=sys.stderr)
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
