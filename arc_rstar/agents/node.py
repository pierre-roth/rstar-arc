import logging
import math
import os
import numpy as np

from arc_rstar.tools.python_tool import extract_python_code
from config import Config, CODE_END, TERMINAL_CODE_END, TERMINAL_MAX_DEPTH, TERMINAL_INVALID
from arc_rstar.arc_task.task import ARCTask

logger = logging.getLogger(__name__)


class Node:
    """
    Node class representing a state in the search tree.
    Supports both Beam Search and MCTS algorithms.
    """

    def __init__(self, config: Config):
        self.config: Config = config
        self.state = {"text": "", "extra_info": ""}
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.depth: int = 0
        self.reward: float = 0
        self.tag = "0"

        self.inited = False

        self.value = 0
        self.__visit_count: int = 0
        self.__value_sum: float = 0

        self.prior_probability: float = 1.0  # Prior probability from policy model (used for PUCT)

        self.terminal_reason: str | None = None  # Reason for terminal node (if applicable)
        self.valid: bool | None = None  # Will be set to True/False when validated
        self.passes_training: bool = False  # Whether the node passes training examples

        self.task: ARCTask | None = None  # Reference to the task used for validation

    def has_children(self) -> bool:
        """Check if the node has any children."""
        return len(self.children) > 0

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

    def add_child(self, text: str) -> "Node":
        """
        Add a child node to this node.

        Args:
            :param text:
        """
        child = Node(self.config)
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        child.tag = f"{self.tag}.{len(self.children) - 1}"
        child.task = self.task
        child.state["text"] = text

        # Validate the child node upon creation
        child.is_valid()

        return child

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update_visit_count(self, count: int) -> None:
        self.__visit_count = count

    def update(self, value: float) -> None:
        if self.inited is False:
            self.inited = True
            self.value = value
        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value: float, start_node: "Node") -> None:
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def puct(self) -> float:
        if not self.parent:
            return 0
        q_value = self.q_value() if self.visit_count() > 0 else 0
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = self.config.c_puct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
        return q_value + u_value

    def is_valid(self) -> bool:
        """
        Check if the node is valid.

        If the node has been previously validated, returns the cached result.
        Otherwise, validates the node and caches the result.
        """
        if self.valid is not None:
            return self.valid

        logger.debug(f"Validating node at depth {self.depth} (terminal: {self.is_terminal()})")

        try:
            # Try to extract the code - for non-terminal nodes this might fail
            logger.debug("Attempting to extract code from node...")

            code = extract_python_code(self.collect_partial_solution())

            logger.debug(f"Successfully extracted code ({len(code.splitlines())} lines)")
            logger.debug("Validation: testing for errors while running training examples")

            # Just check if execution works without errors
            # The function returns (bool, list) but we only check if it returns not None
            result = self.task.run_training_examples(code)
            self.valid = result is not None

            if not self.valid:
                logger.debug("Error detected while running training examples - node is invalid")
            else:
                logger.debug("No errors detected while running training examples - node is valid")

        except Exception as e:
            logger.exception(f"Node validation failed: {str(e)}")
            self.valid = False

        if not self.valid:
            self.terminal_reason = TERMINAL_INVALID

        return self.valid

    def collect_partial_solution(self) -> str:
        # from leaf to root, and reverse
        node = self
        trajectory = []
        while node:
            trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))
