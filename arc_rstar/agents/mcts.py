from arc_rstar.agents import BaseTree
from config import Config


class MCTS(BaseTree):
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config


