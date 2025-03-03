from .tree import BaseTree
from typing import Callable

class BeamSearch(BaseTree):
    def __init__(self):
        super().__init__()
        self.NODE_KEYS: list[str] = ["action", "action_input", "final_answer"]
        self.prompt_wrap: Callable[[...], str] | None = None
        self.obs_wrap: Callable[[str], str] | None = None
        self.step_unwrap: Callable[[...], Dict[str, str]] | None = None
        self.current_top_num: int = 1
        self.current_nodes: list[Type[BaseNode]] = []
        self.final_answer_nodes: List[Type[BaseNode]] = []
        self.candidate_nodes: List[Type[BaseNode]] = []
        self.rollout_idx: int = 0