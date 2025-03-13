from run_tasks import *
from config import Config


if __name__ == '__main__':
    # Create config from command line arguments
    config = Config.from_args()

    if config.all_tasks:
        run_all_tasks(config)
    else:
        run_single_task(config)

    print("Done!")

