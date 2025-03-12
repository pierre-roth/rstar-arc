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
        # self.preference.init_llm()  # Initialize the preference model

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
        
        For beam search, the key configuration parameters are:
        - beam_width: Number of "best candidates" to keep after each step (pruning)
        - branching_factor: Number of candidates to generate at each node (exploration)
        
        Args:
            agent: The search agent to use (e.g., BeamSearch, MCTS)
            task_path: Path to the task JSON file
            
        Returns:
            Result dictionary with solution information
        """

        # Load the task if a path is provided
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
        # Use node's to_dict method if available for JSON serialization
        if hasattr(node, 'to_dict'):
            node_dict = node.to_dict()
        else:
            # Fallback for objects without to_dict method
            node_dict = {
                "final_state": node.state,
                "score": node.score if hasattr(node, "score") else node.value
            }
        
        return node_dict
