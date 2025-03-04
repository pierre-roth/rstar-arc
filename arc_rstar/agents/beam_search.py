from vllm import RequestOutput
from arc_rstar.nodes.base_node import BaseNode
from .tree import BaseTree, code_execution
from arc_rstar.constants import NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS, CODE_END
from typing import Callable, Type, Any


class BeamSearch(BaseTree):
    def __init__(self):
        super().__init__()
        self.NODE_KEYS: list[str] = ["action", "action_input", "final_answer"]
        self.prompt_wrap: Callable[[...], str] | None = None
        self.obs_wrap: Callable[[str], str] | None = None
        self.step_unwrap: Callable[[...], dict[str, str]] | None = None
        self.current_top_num: int = 1
        self.current_nodes: list[Type[BaseNode]] = []
        self.final_answer_nodes: list[Type[BaseNode]] = []
        self.candidate_nodes: list[Type[BaseNode]] = []
        self.rollout_idx: int = 0

    def is_ignored_node(self, node: BaseNode) -> bool:
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

    def create_prompt(
            self,
            is_value_only: bool = False,
    ) -> str:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and self.is_ignored_node(current_node):
                continue
            partial_solution = self.collect_partial_solution(current_node)
            prompt = self.prompt_wrap(
                self.question,
                partial_solution,
                self.config
            )
            if is_value_only:
                prompt = {
                    "prefix": "",
                    "text": prompt,
                }
            prompts.append(prompt)
        return prompts

    @staticmethod
    def is_valid_final_answer_node(node: BaseNode) -> bool:
        if node.is_terminal and node.state["final_answer"] and \
                node.state["final_answer"] not in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            return True
        return False

    def select_next_step(self, outputs=None, from_root=False) -> None:
        self.current_nodes = []
        if outputs is not None:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                candidate_node.value = output.value_estimate if output.value_estimate is not None else 0

        self.candidate_nodes = sorted(self.candidate_nodes, key=lambda x: x.value, reverse=True)
        self.current_nodes = self.candidate_nodes[:]

        for current_node in self.current_nodes[:]:
            if self.__class__.is_valid_final_answer_node(current_node):
                self.final_answer_nodes.append(current_node)
                self.current_nodes.remove(current_node)
                self.current_top_num -= 1
            elif current_node.is_terminal or current_node.depth > self.config.max_depth:
                self.current_nodes.remove(current_node)
        self.current_nodes = self.candidate_nodes[:self.current_top_num]

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            self.current_node = current_node
            for idx, output in enumerate(output.outputs):
                if not output.stop_reason: output.stop_reason = ""
                step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
                self.create_child(step_result, parser_result, current_node)
            self.candidate_nodes.extend(current_node.children)

    def create_node(self, parent: BaseNode | None = None) -> BaseNode:
        return BaseNode(
            parent=parent,
            additional_state_keys=self.NODE_KEYS,
        )

    def create_child(
            self,
            step_result: str,
            parser_result: dict[str, str],
            node: Type[BaseNode],
    ) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                if "Error" in observation:
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
            else:
                new_node.state["text"] = step_result

            if "error" in observation.lower():
                observation = self.obs_wrap(observation)
                step_result = step_result + CODE_END if CODE_END not in step_result else step_result
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                new_node.is_terminal = True
                new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS

        else:
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS

        node.children.append(new_node)

    def get_steps(self):
        final_answer_states = []
        for cur_node in self.final_answer_nodes:
            states = {
                "question": self.question,
                "ground_truth": self.ground_truth,
                "value": cur_node.value,
                "final_answer": cur_node.state["final_answer"],
                "solution": self.collect_partial_solution(cur_node),
                "tag": cur_node.tag,
            }
            final_answer_states.append(states)

        solutions = sorted(final_answer_states, key=lambda x: x['value'], reverse=True)
        return solutions

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




