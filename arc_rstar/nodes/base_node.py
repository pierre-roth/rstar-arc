class BaseNode:

    def __init__(self):
        self.state = {"text": "", "extra_info": ""}
        self.parent = None
        self.children = []
        self.depth = 0
        self.is_terminal = False
        self.reward = 0
        self.value = 0
        self.tag = "0"
        self.consecutive_errors = 0

    def has_children(self) -> bool:
        return self.children != []

    def is_root(self) -> bool:
        return self.parent is None

