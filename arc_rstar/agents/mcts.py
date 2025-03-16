from typing import Optional

from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms.policy import PolicyModel
from arc_rstar.llms.reward import RewardModel
from arc_rstar.tools.python_tool import extract_python_code
from config import Config
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

    def initialize_root(self, prompt: str, task: ARCTask):
        """Initialize the root node with the given prompt."""
        self.root = Node(self.config)
        self.root.state["text"] = prompt
        self.root.task = task

        if self.config.verbose:
            # Just print the string representation which is already JSON (for the visualizer)
            print(f"Added child node: {self.root}")

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

        for child in filter(lambda c: c.valid(), node.children):
            score = child.puct_score()
            if score > best_score:
                best_score = score
                best_child = child

        # If no best child found, return the current node
        if best_child is None:
            return node

        # Recursively select from the best child
        return self.select(best_child)

    def expand(self, node: Node, policy_model: PolicyModel, reward_model: RewardModel) -> list[Node]:
        """
        Expand a node by generating its children.

        Uses the policy model to generate potential next steps and validates them.
        """
        return node.generate_children(policy_model, reward_model)

    def simulate(self, node: Node) -> float:
        """
        Simulation step to estimate the value of a node.

        For terminal nodes: Checks if it solves the task
        For non-terminal nodes: Uses the reward model's score
        """
        # For terminal nodes, check if it solves the task
        if node.is_terminal():
            try:

                # First check if the node is valid
                if not node.valid():
                    return -1.0

                code = extract_python_code(node.get_text(), self.config.verbose)
                success, _ = self.root.task.run_training_examples(code)
                # potentially also check if the code runs successfully on test examples
                return 1.0 if success else -1.0
            except Exception:
                return -1.0

        # For non-terminal nodes, use the reward model's score
        return node.reward

    def backpropagate(self, node: Node, value: float):
        """
        Update node statistics by backpropagation.

        Updates the statistics of the node and all its ancestors.
        """
        node.update_recursive(value)

    def solve(self, task: ARCTask, policy_model: PolicyModel, reward_model: RewardModel) -> Optional[str]:
        """
        Run MCTS to find a solution for the task.

        Performs multiple MCTS simulations and returns the best working solution found.
        If multiple working solutions are found, returns the shortest one.
        If no working solution is found, returns None.
        """
        prompt = get_prompt(self.config, task)
        self.initialize_root(prompt, task)

        if self.config.verbose:
            print(f"Starting MCTS for task: {task.name}")
            print(
                f"C_PUCT: {self.config.c_puct}, Max depth: {self.config.max_depth}, Num simulations: {self.config.num_simulations}")
            print(f"\n\nInitial prompt: {prompt} \n\n\n")

        # Keep track of all working solutions
        working_solutions = []

        # Run simulations
        for sim in range(self.config.num_simulations):
            if self.config.verbose:
                print(f"\n--- Simulation {sim + 1}/{self.config.num_simulations} ---")

            # 1. Selection: Find the most promising leaf node
            selected_node = self.select(self.root)

            if self.config.verbose:
                print(f"Selected node {selected_node.tag} at depth {selected_node.depth}")

            # 2. Expansion: Generate children for the selected node
            expanded_nodes = []
            if not selected_node.is_terminal() and selected_node.depth < self.config.max_depth:
                expanded_nodes = self.expand(selected_node, policy_model, reward_model)

                if self.config.verbose:
                    print(f"Expanded! {len(expanded_nodes)} nodes added")

            # If nodes were expanded, select one for simulation
            if expanded_nodes:
                # Choose the expanded node with the highest immediate reward
                selected_node = max(expanded_nodes, key=lambda x: x.reward)

                if self.config.verbose:
                    print(f"Selected expanded node {selected_node.tag} with reward {selected_node.reward}")

            # 3. Simulation: Estimate the value of the selected node
            value = self.simulate(selected_node)

            if self.config.verbose:
                print(f"Simulated value: {value}")

            # 4. Backpropagation: Update statistics for nodes in the path
            self.backpropagate(selected_node, value)

            # Check if we've found a working solution
            if selected_node.is_terminal() and selected_node.valid():
                try:
                    code = extract_python_code(selected_node.get_text(), self.config.verbose)
                    success, _ = task.run_test_examples(code)

                    if success:
                        # Add to working solutions
                        working_solutions.append(code)
                        if self.config.verbose:
                            print(f"Solution found at node {selected_node.tag}")
                            print(f"Total working solutions found: {len(working_solutions)}")

                    elif self.config.verbose:
                        print(f"Terminal node {selected_node.tag} passes training examples but fails test examples")

                except Exception as e:
                    if self.config.verbose:
                        print(f"Error extracting code from terminal node: {str(e)}")

        # If we found any working solutions, return the shortest one
        if working_solutions:
            if self.config.verbose:
                print(f"\nFound {len(working_solutions)} working solutions!")

            # Find the shortest solution by counting lines of code
            shortest_solution = min(working_solutions, key=lambda code: len(code.splitlines()))

            if self.config.verbose:
                print(f"Selected shortest solution with {len(shortest_solution.splitlines())} lines of code")

            return shortest_solution

        # No working solutions found
        if self.config.verbose:
            print("\nNO WORKING SOLUTION FOUND")

        return None
