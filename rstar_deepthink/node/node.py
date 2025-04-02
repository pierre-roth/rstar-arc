import logging
import time

import numpy as np

from constants import CODE_END, TERMINAL_CODE_END, TERMINAL_MAX_DEPTH, TERMINAL_INVALID
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.tools import run_examples
from rstar_deepthink.tools.python_tool import remove_markers

logger = logging.getLogger(__name__)


class Node:
    """
    Node class representing a state in the search tree.
    Supports both Beam Search and MCTS algorithms.
    """

    def __init__(self, config: Config):
        self.config: Config = config

        self.state = {"prompt_prefix": "", "example_prompt": "", "prompt_suffix": "", "code": ""}
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.example_name: str | None = None  # Name of the example used to generate this node
        self.temperature: float | None = None  # Temperature used to generate this node
        self.depth: int = 0
        self.tag = "0"

        self.inited = False

        self.value = 0
        self.visit_count: int = 0
        self.value_sum: float = 0

        self.terminal: bool | None = None  # Will be set to True/False when terminal
        self.terminal_reason: str | None = None  # Reason for terminal node (will be set when terminal)
        self.valid: bool | None = None  # Will be set to True/False when validated
        self.passes_training: bool | None = None  # Whether the node passes training examples

        self.execution_outputs: list = []  # Store execution outputs for debugging

        self.task: ARCTask | None = None  # Reference to the task used for validation

    def has_children(self) -> bool:
        """Check if the node has any children."""
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""

        if self.terminal is not None:
            return self.terminal

        self.terminal = False

        # Check if maximum depth reached
        if self.depth >= self.config.max_depth:
            self.terminal_reason = TERMINAL_MAX_DEPTH
            self.terminal = True

        # Check if ended with code end marker
        if self.state["code"].strip().endswith(CODE_END):
            self.terminal_reason = TERMINAL_CODE_END
            self.terminal = True

        if not self.valid:
            self.terminal_reason = TERMINAL_INVALID
            self.terminal = True

        return self.terminal

    def add_child(self, code: str, current_temperature: float, current_example: str) -> "Node":
        """
        Add a child node to this node.

        Args:
            :param code:
            :param current_temperature:
            :param current_example:
        """
        child = Node(self.config)

        child.parent = self
        child.depth = self.depth + 1
        child.tag = f"{self.tag}.{len(self.children)}"
        child.task = self.task
        child.state["code"] = code
        child.temperature = current_temperature
        child.example_name = current_example

        # Validate the child node upon creation

        validation_start_time = time.time()

        child.is_valid()

        validation_duration = time.time() - validation_start_time

        logger.debug(f"Validation duration for child {child.tag}: {validation_duration}")

        self.children.append(child)
        logger.debug(f"Added child node {child.tag} to tree.")

        return child

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def update_visit_count(self, count: int) -> None:
        self.visit_count = count

    def update(self, value: float) -> None:
        if self.inited is False:
            self.inited = True
            self.value = value
        self.visit_count += 1
        self.value_sum += value

    def update_recursive(self, value: float) -> None:
        self.update(value)
        if self.parent is not None:
            self.parent.update_recursive(value)

    def puct(self) -> float:
        if not self.parent:
            return 0
        q_value = self.q_value() if self.visit_count > 0 else 0
        if self.parent.visit_count == 0 or self.visit_count == 0:
            u_value = 0
        else:
            u_value = self.config.c_puct * np.sqrt(np.log(self.parent.visit_count) / self.visit_count)
        return q_value + u_value

    def is_valid(self) -> bool:
        """
        Check if the node is valid.

        If the node has been previously validated, returns the cached result.
        Otherwise, validates the node and caches the result.
        """
        if self.valid is not None:
            return self.valid

        logger.debug(f"Validating node at depth {self.depth}.")

        try:
            # Try to extract the code - for non-terminal nodes this might fail
            logger.debug("Attempting to extract code from node...")

            code = remove_markers(self.collect_code())

            logger.debug(f"Successfully extracted code ({len(code.splitlines())} lines)")
            logger.debug("Validating by testing for errors by testing on input grids ...")

            error, passed, self.execution_outputs = run_examples(self.task, code)

            if error:
                self.valid = False
            else:
                self.valid = True
                self.passes_training = passed

        except Exception as e:
            logger.error(f"Node validation failed: {str(e)}")
            self.valid = False

        # running is_terminal() to set terminal_reason
        self.is_terminal()

        logger.debug(f"Node {self.tag} validation successful: valid: {self.valid}, terminal: {self.terminal}")

        return self.valid

    def collect_prompt_and_code(self) -> str:
        # from leaf to root, and reverse
        node = self
        trajectory = []
        while node:
            trajectory.append(
                node.state["prompt_prefix"] + node.state["example_prompt"] + node.state["prompt_suffix"] + node.state[
                    "code"])
            node = node.parent

        return "".join(reversed(trajectory))

    def collect_code(self):
        # Collect code from the node and its ancestors
        node = self
        code = []
        while node:
            code.append(node.state["code"])
            node = node.parent

        return "".join(reversed(code)).strip()

    def collect_metadata(self) -> dict:
        """Calculate the average Q-value of the node."""
        q_values = []
        node = self
        while node:
            q_values.append(node.q_value())
            node = node.parent

        examples_used = []
        node = self
        while node:
            examples_used.append(node.example_name)
            node = node.parent

        temperatures = []
        node = self
        while node:
            temperatures.append(node.temperature)
            node = node.parent

        return {
            "q_values": q_values,
            "examples": examples_used,
            "temperatures": temperatures
        }

    def is_valid_final_answer_node(self) -> bool:
        if self.is_terminal() and self.passes_training:
            return True
        return False
