import logging
from random import choice

from vllm.outputs import RequestOutput

from constants import CODE
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.node import Node
from rstar_deepthink.prompt import get_base_prompt, get_example_prompt

logger = logging.getLogger(__name__)


class Agent:

    def __init__(self, config: Config, task: ARCTask):
        self.config: Config = config

        self.task = task
        self.root: Node | None = None
        self.current_nodes: list[Node] = []
        self.candidate_nodes: list[Node] = []
        self.final_answer_nodes: list[Node] = []
        self.rollout_idx: int = 0
        self.current_temperature: float = self.config.policy_temperature
        self.example_name: str | None = None

        self.create_root(get_base_prompt(config, task), f"{CODE}\ndef solve(I):\n    ", task)

    def create_root(self, base_prompt: (str, str), code: str, task: ARCTask):
        """Initialize the root node with the given state."""
        self.root = Node(self.config)
        self.root.state["prompt_prefix"] = base_prompt[0]
        self.root.state["prompt_suffix"] = base_prompt[1]
        self.root.state["code"] = code
        self.root.task = task
        self.root.valid = True

        self.candidate_nodes.append(self.root)

        logger.debug(
            f"Created root for task {task.name} with prompt: \n{base_prompt[0]}\n<example_will_be_added_later>\n{base_prompt[1]}\n{code}")

    def update(self, rollout_idx: int, current_temperature: float) -> None:
        """Set the example for the root node."""
        self.rollout_idx = rollout_idx
        self.current_temperature = current_temperature

        if self.config.rotate_example:  # Rotate example
            self.example_name = self.config.example_names[self.rollout_idx % len(self.config.example_names)]
            self.root.state["example_prompt"] = get_example_prompt(self.config, self.example_name)
        else:
            self.example_name = choice(self.config.example_names)
            self.root.state["example_prompt"] = get_example_prompt(self.config, self.example_name)

        # Set the example name for the root node
        logger.debug(f"Update root for task {self.task.name} with example: {self.example_name}")
        logger.debug(f"Current temperature: {self.current_temperature}")
        logger.debug(f"Current example prompt: \n{self.root.state['example_prompt']}")

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

    def create_prompts(self, is_value_only: bool = False) -> list[str]:
        """
        if is_value_only, the prompt is used to produce value estimate.
        """
        prompts = []
        current_nodes = self.candidate_nodes if is_value_only else self.current_nodes
        for current_node in current_nodes:
            if not is_value_only and current_node.is_terminal():
                continue
            prompt = current_node.collect_prompt_and_code()
            prompts.append(prompt)

        return prompts

    def select_next_step(self, scores: list[float] | None = None, from_root=False) -> None:
        """Choose which nodes to further develop given newly generated ones"""
        pass

    def generate_next_step(self, outputs: list[RequestOutput]) -> None:
        """Generate and add child nodes from model outputs."""
        self.candidate_nodes = []

        # For each current node, expand with corresponding outputs
        for current_node, request_output in zip(self.current_nodes, outputs):
            if self.config.log_level == "DEBUG":
                if request_output.prompt_token_ids is not None:
                    prompt_token_count = len(request_output.prompt_token_ids)
                else:
                    prompt_token_count = "N/A"

                logger.debug(
                    f"Expanding node at depth {current_node.depth} with {len(request_output.outputs)} children")
                logger.debug(f"Prompt token count: {prompt_token_count}")

            # Create children from outputs
            new_children = []
            # Create children from outputs and log token counts for each generation
            for output in request_output.outputs:
                # Only count if log level is debug:
                if self.config.log_level == "DEBUG":
                    if output.token_ids is not None:
                        child_token_count = len(output.token_ids)
                    else:
                        child_token_count = "N/A"
                    logger.debug(f"Generated child with token count: {child_token_count}")

                # Validation happens here when add_child is called
                child = current_node.add_child(output.text, self.current_temperature, self.example_name)
                new_children.append(child)

            # Add all new children to candidate nodes for evaluation
            self.candidate_nodes.extend(new_children)

        logger.debug(f"Added {len(self.candidate_nodes)} candidate nodes (i.e. children)")
