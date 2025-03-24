import logging

from vllm.outputs import RequestOutput

from arc_rstar.agents.node import Node
from arc_rstar.arc_task.task import ARCTask
from config import Config
from prompt import get_prompt

logger = logging.getLogger(__name__)


class Agent:

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
        """Check if we need to generate for current nodes."""
        if not self.current_nodes:
            logger.debug("No current nodes to generate from")
            return False

        # Check if any current node is non-terminal
        need_generate = any(not node.is_terminal() for node in self.current_nodes)
        logger.debug(f"Need generation: {need_generate} (nodes: {len(self.current_nodes)})")
        return need_generate

    def has_expanded(self) -> bool:
        """Check if current nodes have already been expanded."""
        if not self.current_nodes:
            return False

        # Check if the first current node has children (either all or none have children)
        return self.current_nodes[0].has_children()

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

    def select_next_step(self, scores: list[float] | None = None, from_root=False) -> None:
        """Choose which nodes to further develop given newly generated ones"""
        pass

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        """Take generated outputs and create new nodes"""
        pass
