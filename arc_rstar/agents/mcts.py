import logging
from typing import Any

from vllm.outputs import RequestOutput, CompletionOutput

from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms.policy import PolicyModel
from arc_rstar.llms.reward import RewardModel
from arc_rstar.tools.python_tool import extract_python_code
from config import Config
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
        if step_node.has_children():
            return True
        return False

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
        if from_root:
            start_node = self.root
        else:
            start_node = self.current_nodes[0]

        # select a child node
        node = start_node
        if node is None:
            return None

        next_node = None
        if node.has_children():
            next_node = self.select_child(node)

        return next_node

    def select_child(self, node: Node) -> Node | None:
        best_value = -float("inf")
        best_children = []

        for child in node.children:
            if child.is_terminal():
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_children.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_children = [child]

        # return random.choice(best_children) if best_children else None
        return best_children[0] if best_children else None

    def expand_node(self, outputs: list[CompletionOutput], node: Node) -> None:
        for idx, output in enumerate(outputs):
            node.add_child(output.text)

    def select_next_step(self, scores: list[float] | None = None, from_root=False) -> None:
        self.current_nodes = []
        if scores:
            for candidate_node, score in zip(self.candidate_nodes, scores):
                # backup
                if candidate_node.is_terminal():
                    # for terminal node: update_recursive
                    if not candidate_node.is_valid() or not candidate_node.passes_training:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        candidate_node.update(self.config.positive_reward)
                else:
                    candidate_node.update(score)

                if self.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)

        selection_node = self.selection(from_root=from_root)
        if selection_node is not None:
            self.current_nodes.append(selection_node)

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, request_output in zip(self.current_nodes, outputs):
            self.expand_node(request_output.outputs, current_node)
            self.candidate_nodes.extend(current_node.children)
