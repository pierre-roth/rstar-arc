from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from config import Config
from .node import Node


class Tree(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.root = None
        self.current_node = None
        
    def initialize_root(self, state: Dict[str, Any]) -> Node:
        """Initialize the root node with the given state."""
        self.root = Node()
        self.root.state = state
        self.root.depth = 0
        self.root.tag = "0"
        self.current_node = self.root
        return self.root
        
    @abstractmethod
    def search(self, *args, **kwargs) -> Any:
        """Search algorithm implementation - to be implemented by subclasses."""
        pass
        
    def add_child(self, parent: Node, state: Dict[str, Any]) -> Node:
        """Add a child node to the parent node."""
        child = Node()
        child.state = state
        child.parent = parent
        child.depth = parent.depth + 1
        child.tag = f"{parent.tag}.{len(parent.children)}"
        parent.children.append(child)
        return child

    def get_path_to_node(self, node: Node) -> List[Dict[str, Any]]:
        """Get the path from root to the given node in a serializable format."""
        path = []
        current = node
        while current is not None:
            path.append(current.to_dict())
            current = current.parent
        return path[::-1]  # Reverse to get root-to-node order

