from typing import Any, Optional
import json
import os
from config import Config
from arc_rstar.llms import PolicyModel, ProcessPreferenceModel
from arc_rstar.arc_task.task import ARCTask


class Solver:
    def __init__(self, config: Config):
        self.config = config
        self.policy = PolicyModel(config)
        # remove terminal guidance for preference model
        self.pp = ProcessPreferenceModel(config, terminal_guided=True)
        self.policy.init_llm()  # Initialize the policy model
        self.pp.init_llm()  # Initialize the preference model

    def load_task(self, task_path: str) -> ARCTask:
        """Load an ARC task from the given path."""
        # Create ArcTask from JSON file
        task = ARCTask(task_path, self.config)
        if self.config.verbose:
            print(f"Loaded task: {task_path}")
        return task

    def solve(self, agent, task_path: str = None) -> dict[str, Any]:
        # Load the task if a path is provided
        task = self.load_task(task_path)

        # Run the search algorithm
        if self.config.verbose:
            print(f"Starting {self.config.search_mode} search...")

        final_code = agent.solve(task, self.policy, self.pp)

        # dump agent state
        # agent_state = agent.dump_state()
        if final_code is not None:
            success, outputs = task.run_test_examples(final_code)

            result = {
                "success": success,
                "code": final_code,
                "outputs": outputs
            }

            if self.config.verbose:
                print(f"Search completed! Solution found: {result['success']}")
        else:
            result = {
                "success": False,
                "code": None,
                "outputs": None
            }

        return result
