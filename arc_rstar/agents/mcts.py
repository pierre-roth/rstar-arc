from arc_rstar.agents import Tree
from config import Config


class MCTS:
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config


