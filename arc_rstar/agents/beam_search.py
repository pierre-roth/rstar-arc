from typing import Any, Optional, Tuple
import numpy as np
from config import Config, CODE_END
from arc_rstar.agents.node import Node

from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms.policy import PolicyModel
from arc_rstar.llms.process_preference import ProcessPreferenceModel
from arc_rstar.tools.python_tool import extract_python_code, execute_code_with_grid
from prompt import get_prompt


class BeamSearch:

    def __init__(self, config: Config):
        self.config = config
        self.root = None
        self.beam_width = config.beam_width
        self.branching_factor = config.branching_factor
        self.max_depth = config.max_depth

    def initialize_root(self, prompt: str):
        """Initialize the root node with the given state."""
        self.root = Node(self.config)
        self.root.state["text"] = prompt

    def solve(self, task: ARCTask, policy_model: PolicyModel, pp_model: ProcessPreferenceModel) -> Optional[str]:

        prompt = get_prompt(self.config, task)
        self.initialize_root(prompt)

        if self.config.verbose:
            print(f"Starting beam search for task: {task.name}")
            print(f"Beam width: {self.beam_width}, Branching factor: {self.branching_factor}, Max depth: {self.max_depth}")

        # Initialize beam with just the root node
        beam = [self.root]
        solution_found = False
        solution_node = None

        for depth in range(self.max_depth):
            if not beam:
                if self.config.verbose:
                    print(f"Search stopped: Beam is empty at depth {depth}")
                break

            if self.config.verbose:
                print(f"\n--- Depth {depth+1}/{self.max_depth} ---")
                print(f"Current beam size: {len(beam)}")

            # Generate candidate next steps for all nodes in the current beam
            candidates = []

            for node in beam:
                candidates.extend(node.generate_children(policy_model, pp_model, task))
                
            if self.config.verbose:
                print(f"Total candidates generated: {len(candidates)}")
                
            if not candidates:
                if self.config.verbose:
                    print("Search stopped: No valid candidates generated")
                break

            sorted_candidates = sorted(candidates, key=lambda x: -x.reward)

            # Select top-k candidates for the next beam
            beam = sorted_candidates[:self.beam_width]
            
            if self.config.verbose:
                print(f"New beam size after selection: {len(beam)}")

            # Check if we've found a solution
            for i, node in enumerate(beam):
                if node.is_terminal() and task.run_training_examples(extract_python_code(node.get_text(), self.config.verbose))[0]:
                    solution_found = True
                    solution_node = node
                    if self.config.verbose:
                        print(f"Solution found at depth {depth+1}, beam position {i+1}")
                    break

            if solution_found:
                break

        # If a solution was found, return the code
        if solution_found:
            final_code = extract_python_code(solution_node.get_text(), self.config.verbose)
            if self.config.verbose:
                print("\nSOLUTION FOUND!")
                print(f"Total steps: {solution_node.depth}")
        else:
            final_code = None
            if self.config.verbose:
                print("\nNO SOLUTION FOUND")
                print(f"Search completed after {self.max_depth} steps or beam emptied")

        # Return the best code found
        return final_code
