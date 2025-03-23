import logging

from vllm.outputs import RequestOutput

from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from config import Config
from prompt import get_prompt

logger = logging.getLogger(__name__)


class BS:

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
        need_generate = False
        for step_node in self.current_nodes:
            if not step_node.is_terminal():
                need_generate = True
                break
        return need_generate

    def has_expanded(self) -> bool:
        if not self.current_nodes:
            return False
        step_node = self.current_nodes[0]
        return step_node.has_children()

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

        # Sort all candidates by their value (highest first)
        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)

        # Process terminal nodes: collect successful solutions
        for node in self.candidate_nodes:
            if self.is_valid_final_answer_node(node):
                self.final_answer_nodes.append(node)

        # Keep only non-terminal nodes for expansion
        non_terminal_nodes = [node for node in self.candidate_nodes if not node.is_terminal()]

        # Select top-k non-terminal nodes as the beam for next expansion
        self.current_nodes = non_terminal_nodes[:self.config.beam_width]

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, request_output in zip(self.current_nodes, outputs):
            for i, output in enumerate(request_output.outputs):
                current_node.add_child(output.text)
            self.candidate_nodes.extend(current_node.children)
