import numpy as np
from typing import Dict, List, Optional, Any, Union


class Node:
    def __init__(self, c_puct=2, inited=False, visit_count=0, value_sum=0):
        self.state = {"text": "", "extra_info": ""}
        self.parent = None
        self.children = []
        self.depth = 0
        self.is_terminal = False
        self.reward = 0
        self.value = 0
        self.tag = "0"
        self.consecutive_errors = 0
        self.c_puct = c_puct
        self.inited = False
        self.__visit_count = visit_count
        self.__value_sum = value_sum
        self.score = 0.0  # For beam search
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary for JSON serialization."""
        return {
            "state": self.state,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "reward": self.reward,
            "value": self.value,
            "tag": self.tag,
            "score": self.score,
            "visit_count": self.__visit_count,
            "q_value": self.q_value(),
            # Don't include parent or children to avoid circular references
        }

    def has_children(self) -> bool:
        return self.children != []

    def is_root(self) -> bool:
        return self.parent is None

    def q_value(self) -> float:
        if self.__visit_count == 0:
            return 0
        return self.__value_sum / self.__visit_count

    def visit_count(self) -> int:
        return self.__visit_count

    def update_visit_count(self, count: int) -> None:
        self.__visit_count = count

    def update(self, value: float) -> None:
        if self.inited is False:
            self.inited = True
            self.value = value
        self.__visit_count += 1
        self.__value_sum += value

    def update_recursive(self, value: float, start_node: 'Node') -> None:
        if isinstance(value, list):
            value = float(value[0])
        self.update(value)
        if self.tag == start_node.tag:
            return
        self.parent.update_recursive(value, start_node)

    def puct(self) -> float:
        if not self.parent:
            return 0
        q_value = self.q_value() if self.visit_count() > 0 else 0
        if self.parent.visit_count() == 0 or self.visit_count() == 0:
            u_value = 0
        else:
            u_value = self.c_puct * np.sqrt(np.log(self.parent.visit_count()) / (self.visit_count()))
        return q_value + u_value
        
    def set_score(self, score: float) -> None:
        """Set the score for beam search."""
        self.score = score

