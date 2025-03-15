from typing import Any, Optional, Tuple
import numpy as np
from config import Config, CODE_END
from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms.policy import PolicyModel
from arc_rstar.llms.pp import ProcessPreferenceModel
from arc_rstar.tools.python_tool import extract_python_code
from prompt import get_prompt


class MCTS:
    """
    Monte Carlo Tree Search implementation for solving ARC tasks.

    MCTS is a tree search algorithm that consists of four phases:
    1. Selection: Choose the most promising unexpanded node
    2. Expansion: Generate children of the selected node
    3. Simulation: Estimate the value of the node
    4. Backpropagation: Update statistics up the tree
    """

    def __init__(self, config: Config):
        self.config = config
        self.root = None
        self.c_puct = config.c_puct  # Exploration constant for UCT formula
        self.max_depth = config.max_depth
        self.num_simulations = config.beam_width  # Reusing beam_width as simulation count

    def initialize_root(self, prompt: str, task: ARCTask):
        """Initialize the root node with the given prompt."""
        self.root = Node(self.config)
        self.root.state["text"] = prompt
        self.root.task = task

    def select(self, node: Node) -> Node:
        """
        Select the most promising node to expand using UCT score.

        This recursively traverses the tree until finding a node that either:
        - Is a terminal node
        - Has unvisited children
        - Has no children at all
        """
        # If node is terminal or has no children, return it
        if node.is_terminal() or not node.has_children():
            return node

        # Find the child with the highest UCT score
        best_score = float('-inf')
        best_child = None

        for child in node.children:
            score = child.uct_score(self.c_puct)
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
        """
        Expand a node by generating its children.

        Uses the policy model to generate potential next steps and validates them.
        """
        return node.generate_children(policy_model, pp_model)

    def simulate(self, node: Node, task: ARCTask) -> float:
        """
        Simulation step to estimate the value of a node.

        For terminal nodes: Checks if it solves the task
        For non-terminal nodes: Uses the preference model's score
        """
        # For terminal nodes, check if it solves the task
        if node.is_terminal():
            try:
                # Ensure node has task reference
                if node.task is None:
                    node.task = task
                
                # First check if the node is valid
                if not node.valid():
                    return -1.0
                    
                code = extract_python_code(node.get_text(), self.config.verbose)
                success, _ = task.run_training_examples(code)
                return 1.0 if success else -1.0
            except Exception:
                return -1.0

        # For non-terminal nodes, use the PP model's score
        return node.reward

    def backpropagate(self, node: Node, value: float):
        """
        Update node statistics by backpropagation.

        Updates the statistics of the node and all its ancestors.
        """
        current = node
        while current is not None:
            current.update_stats(value)
            current = current.parent

    def solve(self, task: ARCTask, policy_model: PolicyModel, pp_model: ProcessPreferenceModel) -> Optional[str]:
        """
        Run MCTS to find a solution for the task.

        Performs multiple MCTS simulations and returns the best solution found.
        """
        prompt = get_prompt(self.config, task)
        self.initialize_root(prompt, task)

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
                # Choose the expanded node with the highest immediate reward
                selected_node = max(expanded_nodes, key=lambda x: x.reward)
                
                # Ensure task reference is propagated
                if selected_node.task is None:
                    selected_node.task = task

                if self.config.verbose:
                    print(f"Selected expanded node {selected_node.tag} with reward {selected_node.reward}")

            # 3. Simulation: Estimate the value of the selected node
            value = self.simulate(selected_node, task)

            if self.config.verbose:
                print(f"Simulated value: {value}")

            # 4. Backpropagation: Update statistics for nodes in the path
            self.backpropagate(selected_node, value)

            # Check if we've found a solution
            if selected_node.is_terminal() and selected_node.valid():
                try:
                    code = extract_python_code(selected_node.get_text(), self.config.verbose)
                    success, _ = task.run_training_examples(code)
                    if success:
                        solution_found = True
                        solution_code = code
                        if self.config.verbose:
                            print(f"Solution found at node {selected_node.tag}")
                        break
                    elif self.config.verbose:
                        print(f"Terminal node {selected_node.tag} does not correctly solve the task")
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
            """Recursively search for the best terminal node in the tree."""
            best_node = None
            best_score = float('-inf')

            # Check current node if it's terminal
            if node.is_terminal() and node.visits > 0:
                try:

                    # TODO fix this
                    # Extract code and verify it's valid
                    code = extract_python_code(node.get_text(), self.config.verbose)

                    # Calculate score as a combination of visits and value
                    # This balances exploitation (value) with exploration confidence (visits)
                    score = node.value * (node.visits ** 0.5)

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
                    child_score = child_best.value * (child_best.visits ** 0.5)

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
                # Ensure it's valid
                if not best_terminal.valid():
                    if self.config.verbose:
                        print("Best terminal node is not valid - skipping")
                    return None
                
                code = extract_python_code(best_terminal.get_text(), self.config.verbose)
                # Double check that it correctly solves the task
                success, _ = task.run_training_examples(code)
                if success:
                    if self.config.verbose:
                        print("Best terminal node passes training examples")
                    return code
                else:
                    if self.config.verbose:
                        print("Best terminal node does not pass training examples")
                    # Still return code even if it doesn't pass all examples
                    # This can be useful as a partial solution
                    return code
            except Exception as e:
                if self.config.verbose:
                    print(f"Error extracting code from best terminal node: {str(e)}")

        if self.config.verbose:
            print("\nNO SOLUTION FOUND")

        return None