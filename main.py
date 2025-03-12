import os
import json
import sys
from arc_rstar.agents import BeamSearch, MCTS
from arc_rstar.solver import Solver
from config import Config, SearchMode


def run_single_task(config, task_path=None):
    """Run the solver on a single task."""
    solver = Solver(config)
    
    # Create the appropriate agent based on search_mode
    if config.search_mode == SearchMode.BEAM_SEARCH:
        agent = BeamSearch(config)
    elif config.search_mode == SearchMode.MCTS:
        agent = MCTS(config)
    else:
        print(f"Unknown search mode: {config.search_mode}")
        sys.exit(1)
    
    # Get task path if not provided
    if task_path is None:
        task_path = config.select_task_file()
    
    # Solve the task
    result = solver.solve(agent, task_path)
    
    # Save results
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        task_name = os.path.basename(task_path).split('.')[0] if task_path else f"task_{config.task_index}"
        output_path = os.path.join(config.output_dir, f"{task_name}_{config.search_mode.value}_result.json")
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        if config.verbose:
            print(f"Results saved to {output_path}")
    
    return result


def run_all_tasks(config):
    """Run the solver on all tasks in the specified folder."""
    if config.verbose:
        print(f"Processing all tasks in folder: {config.data_folder}")
    
    files = config.list_task_files()
    
    results = {}
    for i, file_name in enumerate(files):
        if config.verbose:
            print(f"Processing task {i+1}/{len(files)}: {file_name}")
        
        task_file_path = os.path.join(config.data_folder, file_name)
        try:
            result = run_single_task(config, task_file_path)
            results[file_name] = result
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results[file_name] = {"success": False, "error": str(e)}
    
    # Save all results
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"all_tasks_{config.search_mode.value}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if config.verbose:
            print(f"All results saved to {output_path}")
    
    # Print summary
    total = len(results)
    successful = sum(1 for r in results.values() if r.get("success", False))
    print(f"Overall results: {successful}/{total} tasks solved successfully ({successful/total*100:.2f}%)")
    
    return results


if __name__ == '__main__':
    # Check for help flag as a special case
    if "--help" in sys.argv or "-h" in sys.argv:
        Config.print_help()
        sys.exit(0)
        
    # Create config from command line arguments
    config = Config.from_args()

    if config.all_tasks:
        run_all_tasks(config)
    else:
        run_single_task(config)

    print("Done!")