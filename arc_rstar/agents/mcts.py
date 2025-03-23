import logging

from vllm.outputs import RequestOutput
from random import choice

from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from config import Config, TERMINAL_INVALID
from prompt import get_prompt

logger = logging.getLogger(__name__)


class MCTS:

    def __init__(self, config: Config, task: ARCTask):
        self.config: Config = config
        self.task = task
        self.root: Node | None = None
        self.current_nodes: list[Node] = []
        self.candidate_nodes: list[Node] = []
        self.final_answer_nodes: list[Node] = []
        self.max_depth: int = config.max_depth
        self.rollout_idx: int = 0

        self.create_root(get_prompt(config, task), task)

    def create_root(self, prompt: str, task: ARCTask):
        """Initialize the root node with the given state."""
        self.root = Node(self.config)
        self.root.state["text"] = prompt
        self.root.task = task
        self.root.valid = True

        self.candidate_nodes.append(self.root)

    def get_nodes(self) -> list[Node]:
        nodes = []
        candidates = [self.root]
        while candidates:
            node = candidates.pop(0)
            nodes.append(node)
            if node.has_children():
                candidates.extend(node.children)
        return nodes

    def should_generate_next(self) -> bool:
        """Check if we need to generate for current nodes."""
        if not self.current_nodes:
            logger.debug("No current nodes to generate from")
            return False

        # Check if any current node is non-terminal
        need_generate = any(not node.is_terminal() for node in self.current_nodes)
        logger.debug(f"Need generation: {need_generate} "
                     f"(nodes: {len(self.current_nodes)})")
        return need_generate

    def has_expanded(self) -> bool:
        """Check if current nodes have already been expanded."""
        if not self.current_nodes:
            return False

        # Check if all current nodes have children
        return all(node.has_children() for node in self.current_nodes)

    def get_rewards(self):
        rewards = []
        for node in self.current_nodes:
            rewards.append(node.reward if node.reward is not None else 0)  # default reward is 0
        return rewards

    def create_prompts(self, is_value_only: bool = False) -> list[str]:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and current_node.is_terminal():
                continue
            prompt = current_node.collect_partial_solution()
            prompts.append(prompt)

        return prompts

    @staticmethod
    def is_valid_final_answer_node(node: Node) -> bool:
        if node.is_terminal() and node.is_valid() and node.passes_training:
            return True
        return False

    def selection(self, from_root=False) -> Node | None:
        """Select a node for expansion following MCTS tree policy."""
        # Determine starting point for selection
        if from_root:
            current = self.root
        else:
            if not self.current_nodes:
                return None
            current = self.current_nodes[0]

        # If the node is terminal, nothing to select
        if current.is_terminal():
            return None

        # If node has no children, return the node itself for expansion
        if not current.has_children():
            return current

        # Otherwise, select child according to tree policy
        best_child = self.select_child(current)
        if best_child is None:
            # If no valid children, mark node as terminal
            current.terminal_reason = TERMINAL_INVALID
            return None

        # Recursively select from best child
        return best_child if not best_child.is_terminal() else None

    @staticmethod
    def select_child(node: Node) -> Node | None:
        """Select the best child of a node according to PUCT formula."""
        best_value = float("-inf")
        best_children = []

        # Only consider non-terminal children
        valid_children = [child for child in node.children if not child.is_terminal()]
        if not valid_children:
            return None

        for child in valid_children:
            puct_value = child.puct()
            if puct_value > best_value:
                best_value = puct_value
                best_children = [child]
            elif puct_value == best_value:
                best_children.append(child)

        # return a random child if multiple children have the same value
        return choice(best_children) if best_children else None

    def select_next_step(self, scores: list[float] | None = None, from_root=False) -> None:
        """Process evaluations and select next nodes for expansion."""
        # Process candidate nodes if scores are provided
        if scores:
            for candidate_node, score in zip(self.candidate_nodes, scores):
                # Update node statistics
                if candidate_node.is_terminal():
                    # For terminal nodes with solutions
                    if candidate_node.passes_training:
                        candidate_node.update_recursive(self.config.positive_reward)
                    else:
                        candidate_node.update_recursive(self.config.negative_reward)
                else:
                    # For non-terminal nodes
                    candidate_node.update(score)

                # Store valid solutions
                if self.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)

        # Clear current nodes for new selection
        self.current_nodes = []

        # Initialize search from appropriate node
        if from_root:
            self.current_nodes.append(self.root)
        else:
            # Select next node based on tree search
            next_node = self.selection(from_root=False)
            if next_node:
                self.current_nodes.append(next_node)

        # Log current selection state
        logger.debug(f"Selected {len(self.current_nodes)} nodes for next step")

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        """Generate and add child nodes from model outputs."""
        self.candidate_nodes = []

        # For each current node, expand with corresponding outputs
        for current_node, request_output in zip(self.current_nodes, outputs):
            logger.debug(f"Expanding node at depth {current_node.depth} with {len(request_output.outputs)} children")

            # Create children from outputs
            new_children = []
            for output in request_output.outputs:
                child = current_node.add_child(output.text)
                new_children.append(child)

            # Add all new children to candidate nodes for evaluation
            self.candidate_nodes.extend(new_children)

        logger.debug(f"Added {len(self.candidate_nodes)} candidate nodes (i.e. children)")
