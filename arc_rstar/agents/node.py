from config import Config, CODE_END
from arc_rstar.tools.python_tool import extract_python_code, execute_code_with_grid
from arc_rstar.arc_task.task import ARCTask


class Node:

    def __init__(self, config: Config):
        self.config = config
        self.state = {"text": "", "extra_info": ""}
        self.parent = None
        self.children = []
        self.depth = 0
        self.reward = 0
        self.tag = "0"

    def has_children(self) -> bool:
        return self.children != []

    def is_root(self) -> bool:
        return self.parent is None

    def is_terminal(self) -> bool:
        # check whether end of text is CODE_END
        return self.state["text"].strip().endswith(CODE_END)

    def add_child(self, child: "Node"):
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        child.tag = f"{self.tag}.{len(self.children) - 1}"

    # validate nodes based on whether the python code runs
    def valid(self, task: ARCTask) -> bool:

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
                print("Validation: testing for errors while running training examples")

            error = task.run_training_examples(code) is None

            if self.config.verbose:
                if error:
                    print("Error detected while running training examples - node is invalid")
                else:
                    print("No errors detected while running training examples - node is valid")
            return not error

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
            print(f"Generating children for node {self.tag}")

        child_texts = policy_model.generate(prompt)
        if self.config.verbose:
            print(f"Generated {len(child_texts)} candidate continuations")
            for i, child_text in enumerate(child_texts):
                print(f"Child {i + 1}/{len(child_texts)}: {child_text}")

        valid_children = []
        for i, child_text in enumerate(child_texts):
            child = Node(self.config)
            child.state["text"] = child_text
            self.add_child(child)
            child.reward = pp_model.score(child)

            if child.valid(task):
                valid_children.append(child)
                if self.config.verbose:
                    print(f"Child {i + 1}/{len(child_texts)} is valid with reward {child.reward:.4f}")
            else:
                if self.config.verbose:
                    print(f"Child {i + 1}/{len(child_texts)} is invalid and will be discarded")

        if not valid_children and self.config.verbose:
            print("WARNING: No valid children were generated!")
        elif self.config.verbose:
            print(f"Added {len(valid_children)}/{len(child_texts)} valid children")

        return valid_children
