import logging
from random import choice, random

from rstar_deepthink.agents import Agent
from rstar_deepthink.agents.agent_utils import normalized_similarity_score, get_description
from rstar_deepthink.arc_task import Grid
from rstar_deepthink.node import Node

logger = logging.getLogger(__name__)


class Bootstrap(Agent):
    """
        Custom Monte Carlo Tree Search agent that inherits from Agent.
        This leverages shared functionality while maintaining MCTS-specific selection logic.
        In addition, it does hardcoded task object analysis/captioning.
    """
    def __init__(self, config, task):
        super().__init__(config, task)
        self.task_name = task.name
        self.root.state["hint"] = "Here is a hint on how to solve the task: \n" + get_description(self.task_name) + f"\n\nMake sure to write detailed comments for each step!\n\n"

    def should_generate_next(self) -> bool:
        """Check if we need to generate for current nodes."""
        if not self.current_nodes:
            logger.debug("No current nodes to generate from")
            return False

        # Check if any current node is non-terminal
        need_generate = any(not node.is_terminal() for node in self.current_nodes)
        already_solved = len(self.final_answer_nodes) > 2 * self.config.branching_factor
        logger.debug(f"Need generation: {need_generate} (nodes: {len(self.current_nodes)})")
        return need_generate and not already_solved

    def has_expanded(self) -> bool:
        if not self.current_nodes:
            return False

        # Check if the first current node has children (either all or none have children)
        return self.current_nodes[0].has_children() and random() < (1 - 1/(2*self.config.branching_factor))

    @staticmethod
    def select_child(node: Node) -> Node | None:
        """Select the best child of a node according to PUCT formula."""
        best_value = float("-inf")
        best_children = []

        # Only consider non-terminal children
        non_terminal_children = [child for child in node.children if not child.is_terminal()]

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
        if scores:
            for candidate_node, score in zip(self.candidate_nodes, scores):
                # Update node statistics
                if candidate_node.is_terminal():

                    if not candidate_node.is_valid():
                        candidate_node.update_recursive(self.config.negative_reward)
                    else:
                        correct_grids = [example.output_grid for example in self.task.training_examples]
                        predicted_grids = [Grid(output_grid) for output_grid in
                                           candidate_node.execution_outputs[:len(self.task.training_examples)]]

                        candidate_node.update_recursive(normalized_similarity_score(correct_grids, predicted_grids))
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
