import logging
from typing import Optional, List, Tuple

from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms.policy import PolicyModel
from arc_rstar.llms.reward import RewardModel
from arc_rstar.tools.python_tool import extract_python_code
from config import Config, TERMINAL_SUCCESS, TERMINAL_FAILURE, STEP_END, CODE_END
from prompt import get_prompt


class MCTS:
    """
    Monte Carlo Tree Search implementation for solving ARC tasks based on rStar-Math approach.
    """

    def __init__(self, config: Config):
        self.config: Config = config
        self.root: Optional[Node] = None
        self.candidate_nodes: List[Node] = []
        self.final_answer_nodes: List[Node] = []
        self.current_node: Optional[Node] = None

    def initialize_root(self, prompt: str, task: ARCTask):
        """Initialize the root node with the given prompt."""
        self.root = Node(self.config)
        self.root.state["text"] = prompt
        self.root.task = task
        self.current_node = self.root

    def selection(self, from_root: bool = False) -> Optional[Node]:
        """
        Select the most promising node to expand using PUCT.

        If from_root is True, start selection from the root node.
        Otherwise, start from the current node.
        """
        start_node = self.root if from_root else self.current_node

        if start_node is None:
            return None

        if start_node.is_terminal():
            return None

        if start_node.has_children():
            next_node = self.select_child(start_node)
            if next_node is None:  # All children are terminal or invalid
                return None
            return next_node

        return start_node  # Leaf node ready for expansion

    def select_child(self, node: Node) -> Optional[Node]:
        """Select the best child based on PUCT score."""
        best_value = float('-inf')
        best_child = None

        for child in node.children:
            if child.is_terminal() or not child.valid():
                continue

            puct_value = child.puct_score()

            if puct_value > best_value:
                best_value = puct_value
                best_child = child

        return best_child

    def expand(self, node: Node, policy_model: PolicyModel, reward_model: RewardModel) -> List[Node]:
        """
        Generate and validate children for a node.

        Returns list of valid expanded nodes.
        """
        expanded_nodes = node.generate_children(policy_model, reward_model)
        valid_nodes = [child for child in expanded_nodes if child.valid()]

        # Add valid nodes to candidate_nodes list for evaluation
        self.candidate_nodes.extend(valid_nodes)

        return valid_nodes

    def evaluate_terminal_node(self, node: Node) -> float:
        """Evaluate a terminal node by checking if it solves the ARC task."""
        try:
            if not node.valid():
                return -1.0

            code = extract_python_code(node.get_text())
            success, _ = self.root.task.run_training_examples(code)

            if success:
                node.terminal_reason = TERMINAL_SUCCESS
                self.final_answer_nodes.append(node)
            else:
                node.terminal_reason = TERMINAL_FAILURE

            return 1.0 if success else -1.0

        except Exception as e:
            logging.error(f"Error evaluating terminal node: {str(e)}")
            return -1.0

    def evaluate_nodes(self, reward_model: RewardModel):
        """
        Evaluate candidate nodes and backpropagate values.

        Terminal nodes are checked against the task.
        Non-terminal nodes use the reward model's score.
        """
        for node in self.candidate_nodes:
            if node.is_terminal():
                value = self.evaluate_terminal_node(node)
            else:
                value = reward_model.score(node)

            # Backpropagate value up the tree
            node.update(value)

    def select_next_step(self, from_root: bool = False) -> None:
        """Select the next node to expand in the search process."""
        self.current_node = self.selection(from_root)
        self.candidate_nodes = []

        if self.current_node is not None:
            self.candidate_nodes.append(self.current_node)

    def solve(self, task: ARCTask, policy_model: PolicyModel, reward_model: RewardModel) -> Tuple[
              Optional[str], Optional[Node]]:
        """
        Run MCTS to find a solution for the task.

        Performs multiple rollouts and returns the best working solution.
        """
        prompt = get_prompt(self.config, task)
        self.initialize_root(prompt, task)

        logging.info(f"Starting MCTS for task: {task.name}")

        # Run rollouts
        for rollout in range(self.config.num_simulations):

            logging.debug(f"\n--- Rollout {rollout + 1}/{self.config.num_simulations} ---")

            # Start each rollout from the root
            self.select_next_step(from_root=True)

            # Run steps within this rollout until reaching max depth or no nodes to expand
            for step in range(self.config.max_depth):
                logging.debug(f"Rollout {rollout + 1}, Step {step + 1}")

                if not self.current_node:
                    break

                # Expansion phase
                if not self.current_node.is_terminal() and not self.current_node.has_children():
                    expanded_nodes = self.expand(self.current_node, policy_model, reward_model)

                    if expanded_nodes:
                        logging.debug(f"Expanded node {self.current_node.tag}: {len(expanded_nodes)} valid children")

                # Evaluation phase
                self.evaluate_nodes(reward_model)

                # Selection phase for next step
                self.select_next_step()

            logging.debug(f"Rollout {rollout + 1} complete - found {len(self.final_answer_nodes)} potential solutions")

        # After all rollouts, verify solutions on test examples
        working_solutions = []

        for node in self.final_answer_nodes:
            try:
                code = extract_python_code(node.get_text())
                success, _ = task.run_test_examples(code)

                if success:
                    working_solutions.append((code, node))

            except Exception as e:
                logging.error(f"Error testing solution: {str(e)}")

        # Return the best working solution (if any)
        if working_solutions:
            logging.info(f"\nFound {len(working_solutions)} working solutions!")

            # Find the shortest solution by code length
            shortest_solution = min(working_solutions, key=lambda t: len(t[0].replace(CODE_END, '').replace(STEP_END, '').replace('\n\n', '\n').splitlines()))

            logging.debug(f"Selected shortest working solution")

            return shortest_solution

        # No working solutions found
        logging.info("\nNO WORKING SOLUTION FOUND")

        return None, None
