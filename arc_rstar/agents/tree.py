from arc_rstar.nodes import BaseNode
from abc import abstractmethod


class BaseTree:
    def __init__(self):
        self.config = None
        self.question = ""
        self.ground_truth = None
        self.llm = None
        self.root = None
        self.current_node = None
        self.stop = None
        self.node_max_retry = 5

    @abstractmethod
    def create_node(self, parent: BaseNode = None) -> BaseNode:
        pass

    def create_root(self) -> BaseNode:
        root = self.create_node()
        root.state["extra_info"] = f"question: {self.question}"
        return root

    @staticmethod
    def collect_partial_solution(node: BaseNode) -> str:
        trajectory = []
        while node:
            if node.state['text']:
                trajectory.append(node.state['text'])
            node = node.parent
        return "".join(reversed(trajectory))

    def return_states(self) -> dict[str, dict[str, str]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            if node.has_children():
                candidates.extend(node.children)
        return states


def code_execution():
    pass


def code_run():
    pass

