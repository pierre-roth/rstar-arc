import logging
from typing import Any

from vllm.outputs import RequestOutput

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
        self.current_node: Node | None = None
        self.beam_width: int = config.beam_width
        self.branching_factor: int = config.branching_factor
        self.max_depth: int = config.max_depth

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
    def is_valid_final_answer_node(node: Node) -> bool:
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

    def create_node(self, parent: Node | None = None) -> Node:
        return Node(parent=parent, additional_state_keys=self.NODE_KEYS)

    def create_child(self, step_result: str, parser_result: dict[str, str], node: Node) -> None:
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

    def selection(self, from_root=False) -> Optional[Type[MCTSNode]]:
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

    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
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

    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        for idx, output in enumerate(outputs):
            if not output.stop_reason: output.stop_reason = ""
            step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
            self.create_child(step_result, parser_result, node, idx)

    def create_child(self, step_result: str, parser_result: Dict[str, str], node: Type[MCTSNode], idx: int) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            else:
                new_node.state["text"] = step_result

            if "error" in observation.lower():
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors >= self.config.errors_threshold:
                    observation = self.obs_wrap(observation)
                    step_result = step_result + CODE_END if CODE_END not in step_result else step_result
                    new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        else:
            new_node.state["text"] = step_result

        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            # if the final answer is not valid, update the node with negative reward
            node.update(self.config.negative_reward)
            return

        if self.config.is_sampling:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            # just append the node to candidate_nodes, will update the value in select_next_step()
            self.candidate_nodes.append(node)

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
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        # save intermediate metric
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"],
                                                        value_estimate=value_estimate)

                        candidate_node.update_recursive(value_estimate, self.root)
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

    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
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
