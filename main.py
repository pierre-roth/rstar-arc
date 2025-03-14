from run_tasks import run_all_tasks, run_single_task
from config import Config

if __name__ == '__main__':
    # Create config from command line arguments
    config = Config.from_args()
    
    # Debug info
    print(f"Task name from config: '{config.task_name}'")
    print(f"Task index from config: {config.task_index}")
    print(f"Data folder: {config.data_folder}")
    print(f"Search mode: {config.search_mode}")

    if config.all_tasks:
        run_all_tasks(config)
    else:
        run_single_task(config)

    print("Done!")
