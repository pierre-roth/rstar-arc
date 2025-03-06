from typing import Any, Dict, List, Optional
import json
import os
from config import Config
from arc_rstar.llms import PolicyModel, ProcessPreferenceModel
from arc_rstar.arc_task.task import ArcTask


class Solver:
    def __init__(self, config: Config):
        self.config = config
        self.policy = PolicyModel(config)
        self.preference = ProcessPreferenceModel(config)
        self.policy.init_llm()  # Initialize the policy model

    def load_task(self, task_path: str) -> ArcTask:
        """Load an ARC task from the given path."""
        # Create ArcTask from JSON file
        task = ArcTask(task_path)
        if self.config.verbose:
            print(f"Loaded task: {task_path}")
        return task

    def solve(self, agent, task_path: str = None) -> Dict[str, Any]:
        """
        Solve an ARC task using the provided agent.
        
        Args:
            agent: The search agent to use (e.g., BeamSearch, MCTS)
            task_path: Path to the task JSON file
            
        Returns:
            Result dictionary with solution information
        """
        # Load the task if a path is provided
        if task_path:
            task = self.load_task(task_path)
        else:
            # Use task from index or task_name in config
            task_path = self.config.data_folder

            if self.config.task_name:
                task_file = f"{self.config.task_name}.json"
                task_path = os.path.join(task_path, task_file)
            else:
                # Get task by index
                from cli import CLI
                files = CLI.list_task_files(task_path)
                task_path = CLI.select_task_file(files, task_path, self.config.task_index)

            task = self.load_task(task_path)

        # Run the search algorithm
        if self.config.verbose:
            print(f"Starting {self.config.search_mode} search...")

        # Execute the search and get the best node
        best_node = agent.search(task, self.policy)

        # Extract solution from the best node
        solution = self._extract_solution(best_node)

        # Verify solution
        if task.is_solved(best_node.state):
            result = {
                "success": True,
                "message": "Task solved successfully!",
                "solution": solution,
                "path": agent.get_path_to_node(best_node)
            }
        else:
            result = {
                "success": False,
                "message": "Failed to solve the task.",
                "solution": solution,
                "path": agent.get_path_to_node(best_node)
            }

        if self.config.verbose:
            print(f"Search completed: {result['message']}")

        return result

    def _extract_solution(self, node) -> Dict[str, Any]:
        """Extract the solution from the node's state."""
        # This is a placeholder - actual implementation would depend on how
        # solutions are represented in your system
        return {
            "final_state": node.state,
            "score": node.score if hasattr(node, "score") else node.value
        }
