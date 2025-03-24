import logging
from random import shuffle

from vllm.outputs import RequestOutput

from arc_rstar.agents.base_agent import Agent
from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from config import Config
from prompt import get_prompt

logger = logging.getLogger(__name__)


class BS(Agent):

    def select_next_step(self, scores: list[float] | None = None, from_root=False) -> None:
        """
        Select the next nodes to expand in the beam search.

        Args:
            scores: Optional scores for candidate nodes
            from_root: If True, restart search from the root node
        """

        # Regular case: process candidate nodes from previous expansion
        if scores is not None:
            # Update candidate nodes with their scores
            for candidate_node, score in zip(self.candidate_nodes, scores):
                candidate_node.value = score

        # shuffle candidate nodes to break ties randomly (sorting is stable)
        shuffle(self.current_nodes)
        # Sort all candidates by their value (highest first)
        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)

        # Process terminal nodes: collect successful solutions
        for node in self.candidate_nodes:
            if node.is_valid_final_answer_node():
                self.final_answer_nodes.append(node)

        # Keep only non-terminal nodes for expansion
        non_terminal_nodes = [node for node in self.candidate_nodes if not node.is_terminal()]

        # Select top-k non-terminal nodes as the beam for next expansion
        self.current_nodes = non_terminal_nodes[:self.config.beam_width]

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
