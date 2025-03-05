from arc_rstar.nodes import BaseNode
from abc import abstractmethod
from config import Config


class BaseTree:
    def __init__(self, config: Config):
        self.config = config


