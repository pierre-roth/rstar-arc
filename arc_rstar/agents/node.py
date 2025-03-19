import logging
import math
import os

from arc_rstar.tools.python_tool import extract_python_code
from config import Config, CODE_END, TERMINAL_CODE_END, TERMINAL_MAX_DEPTH, TERMINAL_INVALID, TERMINAL_FAILURE, TERMINAL_SUCCESS


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

    def puct_score(self) -> float:
        """
        Calculate the PUCT score for MCTS selection.
        PUCT = exploitation + exploration
              = node.value + c_puct * prior * sqrt(parent_visits) / (1 + node.visits)
        """
        if self.visits == 0:
            return 0

        if self.parent is None:
            return self.value  # Root node has no parent for exploration term

        # Use fixed prior probability of 1.0 for now
        prior_p = self.prior_probability

        # PUCT formula: exploitation + exploration
        exploitation = self.value
        exploration = self.config.c_puct * prior_p * math.sqrt(math.log(self.parent.visits)) / (1 + self.visits)
        return exploitation + exploration

    def update(self, value: float):
        self.visits += 1
        # Use incremental average formula to avoid numerical issues
        self.value = self.value + (value - self.value) / self.visits
        if not self.is_root():
            self.parent.update(value)

    def _validate(self) -> bool:
        """
        Internal method to validate node by running code extraction and execution.
        Returns validation result without caching it.
        
        A node is valid if the code can be extracted and executed without errors,
        not necessarily if it solves the task correctly.
        """
        logging.debug(f"\nValidating node at depth {self.depth} (terminal: {self.is_terminal()})")

        try:
            # Try to extract the code - for non-terminal nodes this might fail
            logging.debug("Attempting to extract code from node...")

            code = extract_python_code(self.get_text())

            logging.debug(f"Successfully extracted code ({len(code.splitlines())} lines)")
            logging.debug("Validation: testing for errors while running training examples")

            # Just check if execution works without errors
            # The function returns (bool, list) but we only check if it returns not None
            result = self.task.run_training_examples(code)
            is_valid = result is not None

            if not is_valid:
                logging.debug("Error detected while running training examples - node is invalid")
            else:
                logging.debug("No errors detected while running training examples - node is valid")

            return is_valid

        except Exception as e:
            logging.exception(f"Node validation failed: {str(e)}")
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
        logging.debug(f"Generating children for node {self.tag}")

        child_texts = policy_model.generate(prompt)
        logging.debug(f"Generated {len(child_texts)} candidate continuations")
        for i, child_text in enumerate(child_texts):
            logging.debug(f"Child {i + 1}/{len(child_texts)}: {child_text}")

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
                logging.debug(f"Child {i + 1}/{len(child_texts)} is valid with reward {child.reward:.4f}")
            else:
                # Mark invalid nodes with negative reward and terminal
                child.reward = -1.0
                logging.debug(f"Child {i + 1}/{len(child_texts)} is invalid and will be discarded")

        if not valid_children:
            logging.debug("WARNING: No valid children were generated!")
        else:
            logging.debug(f"Added {len(valid_children)}/{len(child_texts)} valid children")

        return valid_children

    def __str__(self):
        """Return a JSON string representation of the node including state information."""
        import json

        # Create a JSON representation of the node
        node_json = {
            "node": self.tag,
            "data": {
                "depth": self.depth,
                "reward": self.reward,
                "visits": self.visits,
                "value": self.value,
                "prior_probability": self.prior_probability,
                "is_valid": self.is_valid,
                "terminal_reason": self.terminal_reason,
                "has_children": self.has_children(),
                # Include state information
                "state_text": self.state.get("text", ""),
                "state_extra_info": self.state.get("extra_info", "")
            }
        }
        return json.dumps(node_json)

    def tree_string(self):
        """Return a string representation of the tree starting from this node."""
        tree_str = str(self) + "\n"
        for child in self.children:
            tree_str += child.tree_string()
        return tree_str

    def save_to_file(self):
        output_path = os.path.join(self.config.output_dir, f"detailed_logs", f"job_{self.config.job_id}", f"tree.txt")

        with open(output_path, 'w') as f:
            f.write(self.tree_string())
