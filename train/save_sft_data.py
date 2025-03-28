from rstar_deepthink.config import Config
from rstar_deepthink.node import Node


def save_sft(config: Config, nodes: list[Node]):
    for node in nodes:
        if node.is_valid_final_answer_node():
            pass
