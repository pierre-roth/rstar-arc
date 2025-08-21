import logging
from random import choice

# Removed static import of RequestOutput to avoid heavy dependency at module import.

from constants import SFT_SYSTEM_PROMPT, SFT_IN_BETWEEN_PROMPT, CODE_PREFIX, BOOTSTRAP_SYSTEM_PROMPT, \
    BOOTSTRAP_TASK_PROMPT
from rstar_deepthink.arc_task import ARCTask
from rstar_deepthink.config import Config
from rstar_deepthink.node import Node
from rstar_deepthink.prompt import task_to_prompt, get_example_prompt

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

        self.create_root()

        logger.debug(self.root.collect_prompt_and_code())

    def create_root(self):
        """Initialize the root node with the given state."""
        self.root = Node(self.config)

        if not self.config.fine_tuned:
            self.root.state["system_prompt"] = BOOTSTRAP_SYSTEM_PROMPT
            self.root.state["task_prompt"] = BOOTSTRAP_TASK_PROMPT + task_to_prompt(self.task)
        else:
            self.root.state["system_prompt"] = SFT_SYSTEM_PROMPT
            self.root.state["task_prompt"] = task_to_prompt(self.task) + SFT_IN_BETWEEN_PROMPT

        self.root.state["code"] = CODE_PREFIX
        self.root.task = self.task
        self.root.valid = True

        self.candidate_nodes.append(self.root)

    def update(self, rollout_idx: int, current_temperature: float) -> None:
        """Set the example for the root node."""
        self.rollout_idx = rollout_idx
        self.current_temperature = current_temperature

        if not self.config.fine_tuned:
            if self.config.rotate_example:  # Rotate example
                self.example_name = self.config.example_names[self.rollout_idx % len(self.config.example_names)]
                self.root.state["example_prompt"] = get_example_prompt(self.config, self.example_name)
            else:
                self.example_name = choice(self.config.example_names)
                self.root.state["example_prompt"] = get_example_prompt(self.config, self.example_name)
            # Set the example name for the root node
            logger.debug(f"Update root for task {self.task.name} with example: {self.example_name}")
            logger.debug(f"Current example prompt: \n{self.root.state['example_prompt']}")

        logger.debug(f"Current temperature: {self.current_temperature}")

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
            # For value-only scoring with a non-fine-tuned policy and an active reward model,
            # reformat the prompt to the SFT-style minimal format expected by the reward model.
            if is_value_only and self.config.use_reward_model and not self.config.fine_tuned:
                code_only = current_node.collect_code()
                sft_prompt = (
                    SFT_SYSTEM_PROMPT
                    + task_to_prompt(self.task)
                    + SFT_IN_BETWEEN_PROMPT
                    + code_only
                )
                prompts.append(sft_prompt)
            else:
                prompt = current_node.collect_prompt_and_code()
                prompts.append(prompt)

        return prompts

    def select_next_step(self, scores: list[float] | None = None, from_root=False) -> None:
        """Choose which nodes to further develop given newly generated ones"""
        pass

    def generate_next_step(self, outputs) -> None:
        """Generate and add child nodes from model outputs."""
        self.candidate_nodes = []

        # For each current node, expand with corresponding outputs
        for current_node, request_output in zip(self.current_nodes, outputs):
            prompt_token_count = len(request_output.prompt_token_ids)

            logger.debug(f"Expanding node at depth {current_node.depth} with {len(request_output.outputs)} children")
            logger.debug(f"Prompt token count: {prompt_token_count}")

            # Deduplicate outputs by exact text to avoid identical children
            seen_texts = set()
            deduped_outputs = []
            for output in request_output.outputs:
                text = output.text
                if text not in seen_texts:
                    seen_texts.add(text)
                    deduped_outputs.append(output)

            # Create children from deduplicated outputs
            new_children = []
            # Log token counts for each generation
            for output in deduped_outputs:
                child_token_count = len(output.token_ids)

                logger.debug(f"Generated child with token count: {child_token_count}")

                # Validation happens here when add_child is called
                child = current_node.add_child(output.text, self.current_temperature, self.example_name, prompt_token_count + child_token_count)
                new_children.append(child)

            # Add all new children to candidate nodes for evaluation
            self.candidate_nodes.extend(new_children)

        logger.debug(f"Added {len(self.candidate_nodes)} candidate nodes (i.e. children)")
