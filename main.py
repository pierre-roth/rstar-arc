import logging
from datetime import datetime

from arc_rstar.agents import BS, MCTS
from arc_rstar import Solver
from config import Config
from utils import setup_logging, load_tasks, batch, save_nodes

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
    agent = BS if config.search_mode.lower() == "bs" else MCTS

    # process tasks in batches
    for i, task_batch in enumerate(batch(tasks, config.batch_size)):
        logging.info(f"Solving batch {i} of {len(task_batch)} tasks")
        agents = [agent(config, task) for task in task_batch]

        outputs = solver.solve(agents)

        for output in outputs:
            logger.info(
                f"Task {output[0].task.name} passed: {any(node.is_terminal() and node.is_valid() and node.passes_training for node in output)}")

        # save the nodes to separate files for later analysis
        for nodelist in outputs:
            save_nodes(config, nodelist)

    end_time: datetime = datetime.now()

    logging.info(f"DONE! Total time taken: {end_time - start_time}")
