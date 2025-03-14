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
        
        # Check if this is a terminal node (already has CODE_END)
        is_terminal = self.is_terminal()
        
        if self.config.verbose:
            print(f"\nValidating node at depth {self.depth} (terminal: {is_terminal})")
        
        try:
            # Try to extract the code - for non-terminal nodes this might fail
            if self.config.verbose:
                print("Attempting to extract code from node...")
                
            code = extract_python_code(self.get_text(), self.config.verbose)
            
            if self.config.verbose:
                print(f"Successfully extracted code ({len(code.splitlines())} lines)")
                
            # For non-terminal nodes, basic extraction success means it's valid
            if not is_terminal:
                if self.config.verbose:
                    print("Non-terminal node with valid code structure - passed validation")
                return True
                
            # For terminal nodes, we need to verify execution
            if task.training_examples:
                if self.config.verbose:
                    print("Terminal node - testing code execution on first training example")
                    
                example = task.training_examples[0]
                test_input = example.input_grid.grid
                
                # Just check if execution works, not if result is correct
                result = execute_code_with_grid(code, test_input, self.config.verbose)
                
                if result is not None:
                    if self.config.verbose:
                        print("Code executed successfully and returned a result")
                    return True
                else:
                    if self.config.verbose:
                        print("Code execution failed (returned None)")
                    return False
                    
            if self.config.verbose:
                print("No training examples available, skipping execution check")
            return True
            
        except Exception as e:
            if self.config.verbose:
                print(f"Node validation failed: {str(e)}")
                import traceback
                print(traceback.format_exc())
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
            for i, child_text in enumerate(child_texts):
                print(f"Child {i+1}/{len(child_texts)}: {child_text}")
        
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
