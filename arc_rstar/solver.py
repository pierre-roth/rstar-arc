from typing import Any

from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms import PolicyModel, RewardModel
from arc_rstar.agents import BeamSearch, MCTS
from config import Config


class Solver:
    def __init__(self, config: Config):
        self.config = config
        self.policy = PolicyModel(config)
        self.reward = RewardModel(config, terminal_guided=True)
        self.policy.init_llm()  # Initialize the policy model
        self.reward.init_llm()  # Initialize the reward model

    def load_task(self, task_path: str) -> ARCTask:
        """Load an ARC task from the given path."""
        # Create ArcTask from JSON file
        task = ARCTask(task_path, self.config)
        if self.config.verbose:
            print(f"Loaded task: {task_path}")
        return task

    def solve(self, agent: BeamSearch | MCTS, task_path: str = None) -> dict[str, Any]:
        # Load the task if a path is provided
        task = self.load_task(task_path)

        # Run the search algorithm
        if self.config.verbose:
            print(f"Starting {self.config.search_mode} search...")

        final_code = agent.solve(task, self.policy, self.reward)

        # Print the final tree if verbose (for visualization)
        if self.config.verbose:
            agent.root.print_tree()

        if self.config.verbose:
            print(f"Search completed! Final code: {final_code}")

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
