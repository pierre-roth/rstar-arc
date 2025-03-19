import logging
import os
from typing import Any

from arc_rstar.agents import BeamSearch, MCTS
from arc_rstar.arc_task.task import ARCTask
from arc_rstar.llms import PolicyModel, RewardModel
from config import Config, CODE_END, STEP_END, TERMINAL_SUCCESS, TERMINAL_FAILURE


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
        logging.info(f"Loaded task: {task_path}")
        return task

    def solve(self, agent: BeamSearch | MCTS, task_path: str = None) -> dict[str, Any]:
        # Load the task if a path is provided
        task = self.load_task(task_path)

        # Run the search algorithm
        logging.info(f"Starting {self.config.search_mode} search...")

        final_code, final_node = agent.solve(task, self.policy, self.reward)

        logging.info(f"Search completed! Final code: {final_code}")

        if final_code is not None:
            success, outputs = task.run_test_examples(final_code)

            result = {
                "success": success,
                "code": final_code,
                "outputs": outputs
            }

            if success:
                if final_node is not None:
                    final_node.terminal_reason = TERMINAL_SUCCESS

                output_path = os.path.join(self.config.output_dir, f"detailed_logs", f"job_{self.config.job_id}",
                                           f"{task.name}_solution.py")

                try:
                    with open(output_path, 'w') as f:
                        f.write(final_code.replace(CODE_END, '').replace(STEP_END, '').replace('\n\n', '\n'))
                except Exception as e:
                    logging.error(f"Error saving Python solution code to {output_path}: {e}")

                logging.info(f"Python solution code saved to {output_path}")
            else:
                if final_node is not None:
                    final_node.terminal_reason = TERMINAL_FAILURE

            logging.info(f"Search completed! Solution found: {result['success']}")
        else:
            result = {
                "success": False,
                "code": None,
                "outputs": None
            }

        # Print the final tree if in debug mode
        if self.config.log_level == logging.DEBUG and final_node is not None:
            logging.debug(f"Saving tree to file...")
            agent.root.save_to_file()
            logging.debug(f"Tree saved to file!")

        return result
