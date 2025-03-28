import logging
from datetime import datetime

from rstar_deepthink import Solver
from rstar_deepthink.agents import BS, MCTS, PWMCTS, SMCTS, Custom
from rstar_deepthink.config import Config
from utils import setup_logging, load_tasks, batch, save_nodes, save_summary

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    start_time: datetime = datetime.now()

    # Create config object
    config = Config()

    # Setup logging
    setup_logging(config)

    # Job start info logging
    logging.info(f"Starting job with {config}")

    # load all tasks from data folder
    tasks = load_tasks(config)

    # instantiate the solver
    solver = Solver(config)

    # select the search agent
    agent = {"bs": BS, "mcts": MCTS, "pwmcts": PWMCTS, "smcts": SMCTS, "custom": Custom}.get(config.search_mode, BS)

    # process tasks in batches
    for i, task_batch in enumerate(batch(tasks, config.batch_size)):
        logging.info(f"Solving batch number {i} consisting of {len(task_batch)} tasks")
        agents = [agent(config, task) for task in task_batch]

        outputs = solver.solve(agents)

        for output in outputs:
            logger.info(
                f"Task {output[0].task.name} training examples passed: {any(node.is_valid_final_answer_node() for node in output)}")

        # TODO: filter out the correct solutions and save each one to a jsonl file (1 row per solution)

        if config.save_for_visualization:
            for nodelist in outputs:
                save_nodes(config, nodelist)

        if config.save_sft_data:
            pass

        # save summary of the batch
        save_summary(config, outputs, i)

    end_time: datetime = datetime.now()

    total_time = end_time - start_time
    policy_init_time = config.model_initialization_times['policy']
    reward_init_time = config.model_initialization_times['reward']

    logging.info(f"DONE! Total time taken: {total_time}")
    logging.info(f"Time taken without model initializations: {total_time - policy_init_time - reward_init_time}")
