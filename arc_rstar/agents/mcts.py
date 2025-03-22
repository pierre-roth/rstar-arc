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

    def get_nodes(self) -> list[Node]:
        nodes = []
        candidates = [self.root]
        while candidates:
            node = candidates.pop(0)
            nodes.append(node)
            if node.has_children():
                candidates.extend(node.children)
        return nodes

    def is_ignored_node(self, node: Node) -> bool:
        return node.is_terminal or node.depth > self.config.max_depth

    def should_generate_next(self) -> bool:
        need_generate = False
        for step_node in self.current_nodes:
            if not self.is_ignored_node(step_node):
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
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            prompt = current_node.collect_partial_solution()
            prompts.append(prompt)

        return prompts

    @staticmethod
    def is_valid_final_answer_node(node: Node) -> bool:
        return node.is_terminal() and node.is_valid() and node.passes_training

    def select_next_step(self, scores: list[float] | None, from_root=False) -> None:
        self.current_nodes = []
        if scores is not None:
            for candidate_node, score in zip(self.candidate_nodes, scores):
                candidate_node.value = score

        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.current_nodes = self.candidate_nodes[:]

        for current_node in self.current_nodes[:]:
            if self.is_valid_final_answer_node(current_node):
                self.final_answer_nodes.append(current_node)
                self.current_nodes.remove(current_node)
            elif current_node.is_terminal():
                self.current_nodes.remove(current_node)

        self.current_nodes = self.candidate_nodes[:self.config.beam_width]

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, request_output in zip(self.current_nodes, outputs):
            for i, output in enumerate(request_output.outputs):
                current_node.add_child(output.text)
            self.candidate_nodes.extend(current_node.children)

    def selection(self, from_root=False) -> Node | None:
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node
        # select a child node
        node = start_node
        if node is None: return None
        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)  # To encourage exploration, select from non-terminal children
            if next_node is None:  # if None，it mean all children are terminal
                node.is_terminal = True
            node = next_node
        return None if (node is None or node.is_terminal) else node

    def select_child(self, node: Node) -> Node | None:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        # return random.choice(best_childs) if best_childs else None
        return best_childs[0] if best_childs else None

    def expand_node(self, outputs: list[CompletionOutput], node: Node) -> None:
        for idx, output in enumerate(outputs):
            if not output.stop_reason: output.stop_reason = ""
            node.add_child(output.text)

    def select_next_step(self, outputs=None, from_root=False) -> None:
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []
        if outputs:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and self.config.is_sampling:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                # backup
                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    # for terminal node: update_recursive
                    if candidate_node.state["final_answer"] in []:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        # save intermediate metric
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"],
                                                        value_estimate=value_estimate)

                        candidate_node.update_recursive(value_estimate)
                else:
                    # for intermediate node: just update the value
                    if self.config.terminal_sample:
                        pass
                    else:
                        candidate_node.update(value_estimate)

                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection(from_root=from_root)
        if selection_node is not None:
            self.current_nodes.append(selection_node)

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            value_estimate = output.value_estimate
            if value_estimate is not None:
                self.expand_node(output.outputs, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True

            if self.config.update_leaf_value:
                # if need update leaf node value, just append the node to candidate_nodes, will update the value in select_next_step()
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node)

    def return_states(self) -> dict[str, Any | dict[str, str]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            if node.has_children():
                candidates.extend(node.children)
        states["solutions"] = self.get_steps()
        return states
