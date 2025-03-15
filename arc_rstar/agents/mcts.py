from typing import Any, Optional, Tuple
import numpy as np
import math
from config import Config, CODE_END
from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms.policy import PolicyModel
from arc_rstar.llms.pp import ProcessPreferenceModel
from arc_rstar.tools.python_tool import extract_python_code
from prompt import get_prompt


class MCTS:
    def __init__(self, config: Config):
        self.config = config
        self.root = None
        self.c_puct = config.c_puct  # Exploration constant for UCT formula
        self.max_depth = config.max_depth
        self.num_simulations = config.beam_width  # Reusing beam_width as simulation count

    def initialize_root(self, prompt: str):
        """Initialize the root node with the given state."""
        self.root = Node(self.config)
        self.root.state["text"] = prompt
        # Initialize MCTS-specific attributes
        self.root.visits = 0
        self.root.value = 0.0

    def uct_score(self, node: Node, parent_visits: int) -> float:
        """Calculate the UCT score for a node.

        UCT = exploitation + exploration
             = node.value + c_puct * sqrt(ln(parent_visits) / node.visits)
        """
        # Ensure node has visits attribute
        if not hasattr(node, 'visits'):
            node.visits = 0

        # Ensure node has value attribute
        if not hasattr(node, 'value'):
            node.value = 0.0

        # If node hasn't been visited, prioritize exploration
        if node.visits == 0:
            return float('inf')

        # UCT formula: exploitation + exploration
        return node.value + self.c_puct * math.sqrt(math.log(parent_visits) / node.visits)

    def select(self, node: Node) -> Node:
        """Select a node to expand using UCT."""
        # If node is terminal or has no children, return it
        if node.is_terminal() or not node.has_children():
            return node

        # Find the child with the highest UCT score
        best_score = float('-inf')
        best_child = None

        for child in node.children:
            score = self.uct_score(child, node.visits)
            if score > best_score:
                best_score = score
                best_child = child

        # If no best child found, return the current node
        if best_child is None:
            return node

        # Recursively select from the best child
        return self.select(best_child)

    def expand(self, node: Node, policy_model: PolicyModel, pp_model: ProcessPreferenceModel, task: ARCTask) -> list[
        Node]:
        """Expand a node by generating its children."""
        # Use the existing node.generate_children method
        children = node.generate_children(policy_model, pp_model, task)

        # Initialize MCTS-specific attributes for each child
        for child in children:
            child.visits = 0
            child.value = 0.0

        return children

    def simulate(self, node: Node, task: ARCTask) -> float:
        """Simulation step to estimate the value of a node."""
        # For terminal nodes, check if it solves the task
        if node.is_terminal():
            try:
                code = extract_python_code(node.get_text(), self.config.verbose)
                success, _ = task.run_training_examples(code)
                return 1.0 if success else -1.0
            except Exception:
                return -1.0

        # For non-terminal nodes, use the PP model's score
        return node.reward

    def backpropagate(self, node: Node, value: float):
        """Update node statistics by backpropagation."""
        current = node
        while current is not None:
            # Ensure node has visits attribute
            if not hasattr(current, 'visits'):
                current.visits = 0

            # Ensure node has value attribute
            if not hasattr(current, 'value'):
                current.value = 0.0

            current.visits += 1
            # Update value as a running average
            current.value = ((current.visits - 1) * current.value + value) / current.visits

            # Move to parent
            current = current.parent

    def solve(self, task: ARCTask, policy_model: PolicyModel, pp_model: ProcessPreferenceModel) -> Optional[str]:
        """Run MCTS to find a solution for the task."""
        prompt = get_prompt(self.config, task)
        self.initialize_root(prompt)

        if self.config.verbose:
            print(f"Starting MCTS for task: {task.name}")
            print(f"C_PUCT: {self.c_puct}, Max depth: {self.max_depth}, Num simulations: {self.num_simulations}")
            print(f"\n\nInitial prompt: {prompt} \n\n\n")

        solution_found = False
        solution_code = None

        # Run simulations
        for sim in range(self.num_simulations):
            if self.config.verbose:
                print(f"\n--- Simulation {sim + 1}/{self.num_simulations} ---")

            # 1. Selection: Find the most promising leaf node
            selected_node = self.select(self.root)

            if self.config.verbose:
                print(f"Selected node {selected_node.tag} at depth {selected_node.depth}")

            # 2. Expansion: Generate children for the selected node
            expanded_nodes = []
            if not selected_node.is_terminal() and selected_node.depth < self.max_depth:
                expanded_nodes = self.expand(selected_node, policy_model, pp_model, task)

                if self.config.verbose:
                    print(f"Expanded {len(expanded_nodes)} nodes")

            # If nodes were expanded, select one for simulation
            if expanded_nodes:
                # Choose the expanded node with highest immediate reward
                selected_node = max(expanded_nodes, key=lambda x: x.reward)

                if self.config.verbose:
                    print(f"Selected expanded node {selected_node.tag} with reward {selected_node.reward}")

            # 3. Simulation: Estimate the value of the selected node
            value = self.simulate(selected_node, task)

            if self.config.verbose:
                print(f"Simulated value: {value}")

            # 4. Backpropagation: Update statistics for nodes in the path
            self.backpropagate(selected_node, value)

            # Check if we've found a solution
            if selected_node.is_terminal():
                try:
                    code = extract_python_code(selected_node.get_text(), self.config.verbose)
                    success, _ = task.run_training_examples(code)
                    if success:
                        solution_found = True
                        solution_code = code
                        if self.config.verbose:
                            print(f"Solution found at node {selected_node.tag}")
                        break
                except Exception as e:
                    if self.config.verbose:
                        print(f"Error extracting code from terminal node: {str(e)}")

        # If a solution was found, return it
        if solution_found:
            if self.config.verbose:
                print("\nSOLUTION FOUND!")
            return solution_code

        # If no solution found during simulations, find the best terminal node
        def find_best_terminal(node):
            best_node = None
            best_score = float('-inf')

            # Check current node if it's terminal
            if node.is_terminal() and hasattr(node, 'visits') and hasattr(node, 'value') and node.visits > 0:
                try:
                    # Extract code (just to make sure it's valid)
                    code = extract_python_code(node.get_text(), self.config.verbose)

                    # Calculate score as a combination of visits and value
                    # This balances exploitation (value) with exploration confidence (visits)
                    score = node.value * math.sqrt(node.visits)

                    if score > best_score:
                        best_node = node
                        best_score = score
                except Exception:
                    pass

            # Check children recursively
            for child in node.children:
                child_best = find_best_terminal(child)
                if child_best is not None:
                    # Get score for child_best
                    child_score = child_best.value * math.sqrt(child_best.visits) if hasattr(child_best,
                                                                                             'value') and hasattr(
                        child_best, 'visits') else float('-inf')

                    if child_score > best_score:
                        best_node = child_best
                        best_score = child_score

            return best_node

        best_terminal = find_best_terminal(self.root)

        if best_terminal is not None:
            if self.config.verbose:
                print(
                    f"\nBest terminal node found: {best_terminal.tag} with value {best_terminal.value} and visits {best_terminal.visits}")
            try:
                code = extract_python_code(best_terminal.get_text(), self.config.verbose)
                # Double check that the code is valid
                success, _ = task.run_training_examples(code)
                if success:
                    if self.config.verbose:
                        print("Best terminal node passes training examples")
                else:
                    if self.config.verbose:
                        print("Best terminal node does not pass training examples")
                return code
            except Exception as e:
                if self.config.verbose:
                    print(f"Error extracting code from best terminal node: {str(e)}")

        if self.config.verbose:
            print("\nNO SOLUTION FOUND")

        return None
