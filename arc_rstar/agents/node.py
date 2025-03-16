import config
from config import Config, CODE_END, TERMINAL_CODE_END, TERMINAL_MAX_DEPTH, TERMINAL_INVALID
from arc_rstar.tools.python_tool import extract_python_code, execute_code_with_grid
from arc_rstar.arc_task.task import ARCTask
import math


class Node:
    """
    Node class representing a state in the search tree.
    Supports both Beam Search and MCTS algorithms.
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = {"text": "", "extra_info": ""}
        self.parent = None
        self.children = []
        self.depth = 0
        self.reward = 0  # Immediate reward (used by both Beam Search and MCTS)
        self.tag = "0"

        # MCTS-specific attributes
        self.visits = 0  # Number of visits to this node
        self.value = 0.0  # Average of simulation values (used for MCTS)
        self.prior_probability = 1.0  # Prior probability from policy model (used for PUCT)

        # terminal reason tracking
        self.terminal_reason = None  # Reason for terminal node (if applicable)

        # Validation status (set during child creation)
        self.is_valid = None  # Will be set to True/False when validated
        self.task = None  # Reference to the task used for validation

    def has_children(self) -> bool:
        """Check if the node has any children."""
        return len(self.children) > 0

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""

        # If terminal reason is already set, return True
        if self.terminal_reason is not None:
            return True

        # Check if ended with code end marker
        if self.state["text"].strip().endswith(CODE_END):
            self.terminal_reason = TERMINAL_CODE_END
            return True

        # Check if maximum depth reached
        if self.depth >= self.config.max_depth:
            self.terminal_reason = TERMINAL_MAX_DEPTH
            return True

        return False

    def add_child(self, child: "Node"):
        """
        Add a child node to this node.

        Args:
            child: The child node to add
        """
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        child.tag = f"{self.tag}.{len(self.children) - 1}"
        child.task = self.task

    def puct_score(self, c_puct: float) -> float:
        """
        Calculate the PUCT score for MCTS selection.
        PUCT = exploitation + exploration
              = node.value + c_puct * prior * sqrt(parent_visits) / (1 + node.visits)
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes

        if self.parent is None:
            return self.value  # Root node has no parent for exploration term

        # Use fixed prior probability of 1.0 for now
        prior_p = self.prior_probability

        # PUCT formula: exploitation + exploration
        exploitation = self.value
        exploration = c_puct * prior_p * math.sqrt(math.log(self.parent.visits)) / (1 + self.visits)
        return exploitation + exploration

    def update_stats(self, simulation_value: float):
        """Update node statistics based on simulation result."""
        self.visits += 1
        # Update value as a running average
        self.value = ((self.visits - 1) * self.value + simulation_value) / self.visits

    def update(self, value: float):
        """Update just this node's statistics."""
        self.visits += 1
        self.value = ((self.visits - 1) * self.value + value) / self.visits

    def update_recursive(self, value: float, stop_node=None):
        """Update this node and all ancestors up to stop_node."""
        current = self
        while current is not None and current is not stop_node:
            current.visits += 1
            current.value = ((current.visits - 1) * current.value + value) / current.visits
            current = current.parent

    def _validate(self) -> bool:
        """
        Internal method to validate node by running code extraction and execution.
        Returns validation result without caching it.
        
        A node is valid if the code can be extracted and executed without errors,
        not necessarily if it solves the task correctly.
        """
        if self.task is None:
            if self.config.verbose:
                print("Cannot validate node: task reference is missing")
            return False

        if self.config.verbose:
            print(f"\nValidating node at depth {self.depth} (terminal: {self.is_terminal()})")

        try:
            # Try to extract the code - for non-terminal nodes this might fail
            if self.config.verbose:
                print("Attempting to extract code from node...")

            code = extract_python_code(self.get_text(), self.config.verbose)

            if self.config.verbose:
                print(f"Successfully extracted code ({len(code.splitlines())} lines)")
                print("Validation: testing for errors while running training examples")

            # Just check if execution works without errors
            # The function returns (bool, list) but we only check if it returns not None
            result = self.task.run_training_examples(code) 
            is_valid = result is not None

            if self.config.verbose:
                if not is_valid:
                    print("Error detected while running training examples - node is invalid")
                else:
                    print("No errors detected while running training examples - node is valid")
            return is_valid

        except Exception as e:
            if self.config.verbose:
                print(f"Node validation failed: {str(e)}")
                import traceback
                print(traceback.format_exc())
            return False

    def valid(self) -> bool:
        """
        Check if the node is valid.

        If the node has been previously validated, returns the cached result.
        Otherwise, validates the node and caches the result.
        """
        # If this node hasn't been validated yet
        if self.is_valid is None:
            self.is_valid = self._validate()

            # If node is invalid, set it as terminal with invalid reason
            if not self.is_valid:
                self.terminal_reason = TERMINAL_INVALID

        return self.is_valid

    # recursively collect all the text up to the root (root text is in front)
    def get_text(self) -> str:
        """Get the complete text of this node, including all parent nodes."""
        text = self.state["text"]
        if not self.is_root():
            text = self.parent.get_text() + "\n" + text
        return text

    def generate_children(self, policy_model, reward_model) -> list["Node"]:
        """Generate and validate children for this node."""
        prompt = self.get_text()
        if self.config.verbose:
            print(f"Generating children for node {self.tag}")

        child_texts = policy_model.generate(prompt)
        if self.config.verbose:
            print(f"Generated {len(child_texts)} candidate continuations")
            for i, child_text in enumerate(child_texts):
                print(f"Child {i + 1}/{len(child_texts)}: {child_text}")

        valid_children = []
        for i, child_text in enumerate(child_texts):
            child = Node(self.config)
            child.state["text"] = child_text

            self.add_child(child)
            child.reward = reward_model.score(child)

            # Validate the child node
            is_valid = child.valid()
            
            if is_valid:
                valid_children.append(child)
                if self.config.verbose:
                    print(f"Child {i + 1}/{len(child_texts)} is valid with reward {child.reward:.4f}")
            else:
                if self.config.verbose:
                    print(f"Child {i + 1}/{len(child_texts)} is invalid and will be discarded")

        if not valid_children and self.config.verbose:
            print("WARNING: No valid children were generated!")
        elif self.config.verbose:
            print(f"Added {len(valid_children)}/{len(child_texts)} valid children")

        return valid_children

