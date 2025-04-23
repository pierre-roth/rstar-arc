import json
import logging
import os

from rstar_deepthink.config import Config
from rstar_deepthink.node import Node
from rstar_deepthink.tools import test_correct  # Keep original test_correct import

logger = logging.getLogger(__name__)


# --- Helper Function to Check Correctness ---
def _is_correct_final_solution(node: Node) -> bool:
    """
    Checks if a node represents a valid, terminal solution that passes test examples.

    Args:
        node: The Node object to check.

    Returns:
        True if the node is a correct final solution, False otherwise.
    """
    # is_valid_final_answer_node checks if it's terminal and passed training examples
    if not node.is_valid_final_answer_node():
        return False

    # Additionally check if it passes test examples
    error, passed_test, _ = test_correct(node)
    return not error and passed_test


# --- Helper Function to Compute Final Outcomes via DFS ---
def _compute_final_outcomes(nodes: list[Node]):
    """
    Computes `final_correct` and `final_wrong` attributes for all nodes in the tree
    using a Depth-First Search (DFS) approach.

    Args:
        nodes: A flat list of all nodes in the MCTS tree for a single task.
               Nodes are modified in-place.
    """
    visited = set()  # To handle potential cycles, though unlikely in standard MCTS

    def dfs(node: Node):
        """Inner recursive DFS function."""
        if node is None or node.tag in visited:
            return
        visited.add(node.tag)

        is_leaf_or_terminal = not node.children or node.is_terminal()

        if is_leaf_or_terminal:
            # For terminal nodes, check correctness
            if node.is_terminal():
                if _is_correct_final_solution(node):
                    node.final_correct = 1
                    # logger.debug(f"Node {node.tag} is correct terminal.")
                else:
                    # Includes invalid terminal nodes and those failing tests
                    node.final_wrong = 1
                    # logger.debug(f"Node {node.tag} is incorrect/invalid terminal.")
            # else:
            #    node.final_wrong = 1 # Or handle differently if needed
        else:
            # For non-terminal nodes, recurse on children first
            for child in node.children:
                dfs(child)
                # Aggregate results from children
                node.final_correct += child.final_correct
                node.final_wrong += child.final_wrong
            # logger.debug(f"Node {node.tag} aggregated: correct={node.final_correct}, wrong={node.final_wrong}")

    # Find root node(s) (those without a parent)
    root_nodes = [node for node in nodes if node.parent is None]
    if not root_nodes and nodes:  # Fallback if parent links weren't set correctly
        root_nodes = [nodes[0]]
        logger.warning("Could not definitively find root node(s), starting DFS from the first node.")

    # Start DFS from root(s)
    for root in root_nodes:
        logger.debug(f"Starting outcome computation DFS from root: {root.tag}")
        dfs(root)


def _extract_solutions(nodes: list[Node], config: Config) -> list[dict]:
    """
    Extracts valid solutions from the MCTS tree nodes.

    Args:
        nodes: A flat list of all nodes in the MCTS tree for a single task.
        config: The configuration object.

    Returns:
        A list of dictionaries, each representing a valid solution.
    """
    task_name = nodes[0].task.name
    solutions = []

    # Iterate through nodes to find valid final answer nodes
    for node in nodes:
        if _is_correct_final_solution(node):
            solution_code = node.collect_code()
            metadata = node.collect_metadata()  # Collects Q-values, examples, temps along path
            solution_data = {
                "task_name": task_name,
                "solution_code": solution_code,
                "metadata": metadata
            }
            solutions.append(solution_data)

    return solutions


# --- Preference Pair Extraction ---
def _extract_preference_pairs(nodes: list[Node], config: Config) -> list[dict]:
    """
    Extracts preference pairs (chosen, rejected) from the MCTS tree nodes.
    Assumes _compute_final_outcomes has been called first.

    Args:
        nodes: A flat list of all nodes in the MCTS tree for a single task.
        config: The configuration object.

    Returns:
        A list of dictionaries, each representing a preference pair.
    """
    task_name = nodes[0].task.name
    preference_pairs = []

    # Iterate through nodes to find decision points
    for node in nodes:
        if not node.is_valid() or not node.children:  # Skip leaves
            continue

        # Separate children based on whether they lead to correct/incorrect outcomes
        # These attributes should now exist after calling _compute_final_outcomes
        chosen_candidates = [child for child in node.children if child.is_valid() and child.final_correct > 0]
        rejected_candidates = [child for child in node.children if child.is_valid() and child.final_wrong > 0]

        # Only proceed if we have both types of candidates
        if not chosen_candidates or not rejected_candidates:
            continue

        # Sort candidates by Q-value
        chosen_candidates.sort(key=lambda x: x.q_value(), reverse=True)
        rejected_candidates.sort(key=lambda x: x.q_value(), reverse=False)  # Lowest Q-value first

        # Select top N (e.g., N=2 as per paper/code)
        num_pairs_to_select = 2
        chosen_nodes = chosen_candidates[:num_pairs_to_select]
        rejected_nodes = rejected_candidates[:num_pairs_to_select]

        # Reconstruct the prefix (only code up to the split point)
        prefix_code = node.collect_code()

        # Create pairs
        for chosen_node in chosen_nodes:
            for rejected_node in rejected_nodes:
                # Basic check: ensure chosen Q > rejected Q
                if chosen_node.q_value() > rejected_node.q_value() + config.min_step_margin:
                    preference_pair_data = {
                        "task_name": task_name,
                        "prefix": prefix_code,  # Prompt + code up to the split point
                        "chosen": chosen_node.state['code'],  # The chosen step's code
                        "rejected": rejected_node.state['code'],  # The rejected step's code
                        "metadata": {
                            "chosen_q": chosen_node.q_value(),
                            "rejected_q": rejected_node.q_value(),
                            "chosen_tag": chosen_node.tag,
                            "rejected_tag": rejected_node.tag,
                            "parent_tag": node.tag,
                            "prefix_final_correct": node.final_correct,
                            "prefix_final_wrong": node.final_wrong,
                            "chosen_final_correct": chosen_node.final_correct,
                            "rejected_final_wrong": rejected_node.final_wrong,
                            "prefix_temperature": node.temperature,
                            "chosen_temperature": chosen_node.temperature,
                            "rejected_temperature": rejected_node.temperature
                        }
                    }
                    preference_pairs.append(preference_pair_data)

    return preference_pairs


# --- Main Saving Function ---
def save_sft_data(config: Config, nodes: list[Node]):
    """
    Save successful task solutions (policy SFT data) and preference pairs (reward model sft data)
    to JSONL files. Computes final outcomes before extracting pairs.

    Args:
        config: The configuration object.
        nodes: List of all nodes in the MCTS tree for a single task.
    """
    if not nodes:
        logger.warning("Received empty node list in save_sft. This should NOT happen! Skipping.")
        return

    # --- Setup Directories and Paths ---
    output_dir = os.path.join(config.sft_data_dir, f"round_{config.round_number}")
    os.makedirs(output_dir, exist_ok=True)

    if not config.evaluation:
        solutions_file_path = os.path.join(output_dir, "solutions_training.jsonl")  # File for SFT data
        preference_pairs_file_path = os.path.join(output_dir,
                                                  "preference_pairs_training.jsonl")  # File for RM preference data
    else:
        solutions_file_path = os.path.join(output_dir, "solutions_evaluation.jsonl")  # File for SFT data
        preference_pairs_file_path = os.path.join(output_dir,
                                                  "preference_pairs_evaluation.jsonl")  # File for RM preference data

    task_name = nodes[0].task.name
    logger.debug(f"Processing save_sft for task: {task_name}")

    # --- 1. Compute Final Outcomes (Correct/Wrong Counts) ---
    logger.debug(f"Computing final outcomes for {len(nodes)} nodes for task {task_name}...")
    _compute_final_outcomes(nodes)
    logger.debug(f"Finished computing final outcomes for task {task_name}.")

    # --- 2. Save Solutions ---
    solutions_to_save = _extract_solutions(nodes, config)

    # Append SFT data to raw.jsonl
    if solutions_to_save:
        try:
            with open(solutions_file_path, 'a', encoding="utf-8") as f:
                for solution in solutions_to_save:
                    entry = json.dumps(solution)
                    f.write(entry + '\n')
            logger.info(f"Saved {len(solutions_to_save)} SFT entries for task {task_name} to {solutions_file_path}")
        except IOError as e:
            logger.error(f"Failed to write SFT data for task {task_name} to {solutions_file_path}: {e}")
    else:
        logger.info(f"No SFT entries to save for task {task_name}.")

    # --- 3. Save Preference Pairs ---
    preference_pairs_to_save = _extract_preference_pairs(nodes, config)

    # Append preference data to preference.jsonl
    if preference_pairs_to_save:
        try:
            with open(preference_pairs_file_path, 'a', encoding="utf-8") as f:
                for preference_pair in preference_pairs_to_save:
                    entry = json.dumps(preference_pair)
                    f.write(entry + '\n')
            logger.info(
                f"Saved {len(preference_pairs_to_save)} preference pairs for task {task_name} to {preference_pairs_file_path}")
        except IOError as e:
            logger.error(f"Failed to write preference data for task {task_name} to {preference_pairs_file_path}: {e}")
    else:
        logger.info(f"No preference pairs to save for task {task_name}.")
