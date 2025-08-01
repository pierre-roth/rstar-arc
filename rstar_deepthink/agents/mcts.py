import logging
from random import choice

from rstar_deepthink.agents import Agent
from rstar_deepthink.node import Node

logger = logging.getLogger(__name__)


class MCTS(Agent):
    """
    Monte Carlo Tree Search agent that inherits from the Beam Search (BS) agent.
    This leverages shared functionality while maintaining MCTS-specific selection logic.
    """

    @staticmethod
    def select_child(node: Node) -> Node | None:
        """Select the best child of a node according to PUCT formula."""
        best_value = float("-inf")
        best_children = []

        # Only consider non-terminal children
        non_terminal_children = [child for child in node.children if not child.is_terminal()]
        if not non_terminal_children:
            return None

        for child in non_terminal_children:
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
        if scores and all(score is not None for score in scores):
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
                if candidate_node.is_valid_final_answer_node():
                    self.final_answer_nodes.append(candidate_node)

        # Initialize search from appropriate node
        if from_root:
            self.current_nodes = [self.root]
        else:
            # Select next node based the current node (there is only one)
            selected = self.select_child(self.current_nodes[0])
            if selected is not None:
                self.current_nodes = [selected]

        # Log current selection state
        logger.debug(f"Selected {len(self.current_nodes)} nodes for next step")
