from .base_node import BaseNode
import numpy as np


class MCTSNode(BaseNode):
    def __init__(self, c_puct=2, inited=False, visit_count=0, value_sum=0):
        super().__init__()
        self.c_puct = c_puct
        self.inited = False
        self.__visit_count = visit_count
        self.__value_sum = value_sum

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

    def update_recursive(self, value: float, start_node: BaseNode) -> None:
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

