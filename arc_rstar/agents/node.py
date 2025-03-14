from config import Config, CODE_END


class Node:
    def __init__(self, config: Config):
        self.config = config
        self.state = {"text": "", "extra_info": ""}
        self.parent = None
        self.children = []
        self.depth = 0
        self.reward = 0

    def has_children(self) -> bool:
        return self.children != []

    def is_root(self) -> bool:
        return self.parent is None

    def is_terminal(self) -> bool:
        return self.get_text().count(CODE_END) > 1

    def add_child(self, child: "Node"):
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1

    # validate nodes based on whether the python code runs
    def valid(self, task) -> bool:
        from arc_rstar.tools.python_tool import extract_python_code, execute_code_with_grid
        
        try:
            # Try to extract the code
            code = extract_python_code(self.get_text())
            
            # Use the first training example to validate
            if task.training_examples:
                example = task.training_examples[0]
                test_input = example.input_grid.grid
                
                # Just check if execution works, not if result is correct
                execute_code_with_grid(code, test_input)
                return True
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"Node validation failed: {str(e)}")
            return False

    # recursively collect all the text up to the root (root text is in front)
    def get_text(self) -> str:
        text = self.state["text"]
        if not self.is_root():
            text = self.parent.get_text() + "\n" + text
        return text

    def generate_children(self, policy_model, pp_model, task) -> list["Node"]:
        prompt = self.get_text()
        if self.config.verbose:
            print(f"Generating children for node at depth {self.depth}")
        
        child_texts = policy_model.generate(prompt)
        if self.config.verbose:
            print(f"Generated {len(child_texts)} candidate continuations")
        
        valid_children = []
        for i, child_text in enumerate(child_texts):
            child = Node(self.config)
            child.state["text"] = child_text
            
            # Check if the node is valid before adding it
            if child.valid(task):
                child.reward = pp_model.score(child)
                self.add_child(child)
                valid_children.append(child)
                if self.config.verbose:
                    print(f"Child {i+1}/{len(child_texts)} is valid with reward {child.reward:.4f}")
            else:
                if self.config.verbose:
                    print(f"Child {i+1}/{len(child_texts)} is invalid and will be discarded")
        
        if not valid_children and self.config.verbose:
            print("WARNING: No valid children were generated!")
        elif self.config.verbose:
            print(f"Added {len(valid_children)}/{len(child_texts)} valid children")
            
        return valid_children
