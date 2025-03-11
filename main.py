import os
import json
import sys
from arc_rstar.agents import BeamSearch, MCTS
from arc_rstar.solver import Solver
from cli import CLI
from config import Config


def run_single_task(config, task_path=None):
    """Run the solver on a single task."""
    solver = Solver(config)
    
    # Create the appropriate agent based on search_mode
    if config.search_mode.lower() == "beam_search":
        agent = BeamSearch(config)
    elif config.search_mode.lower() == "mcts":
        agent = MCTS(config)
    else:
        print(f"Unknown search mode: {config.search_mode}")
        sys.exit(1)
    
    # If task_path is not provided but task_name is, find the task file
    if task_path is None and config.task_name:
        # Find the task in the data folder
        task_file = f"{config.task_name}.json"
        task_path = os.path.join(config.data_folder, task_file)
        
        if not os.path.exists(task_path):
            print(f"Error: Task file '{task_path}' not found.")
            sys.exit(1)
        
        if config.verbose:
            print(f"Using task file: {task_path}")
    elif task_path is None:
        # If no specific task path or name, use task_index to select from data folder
        files = CLI.list_task_files(config.data_folder)
        task_path = CLI.select_task_file(files, config.data_folder, config.task_index, config.verbose)
    
    # Solve the task
    result = solver.solve(agent, task_path)
    
    # Save results
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        task_name = os.path.basename(task_path).split('.')[0] if task_path else f"task_{config.task_index}"
        output_path = os.path.join(config.output_dir, f"{task_name}_{config.search_mode}_result.json")
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    return result


def run_all_tasks(config):
    """Run the solver on all tasks in the specified folder."""
    # Use data folder from config
    task_path = config.data_folder
    
    if config.verbose:
        print(f"Processing all tasks in folder: {task_path}")
    
    files = CLI.list_task_files(task_path)
    
    results = {}
    for i, file_name in enumerate(files):
        if config.verbose:
            print(f"Processing task {i+1}/{len(files)}: {file_name}")
        
        task_file_path = os.path.join(task_path, file_name)
        try:
            result = run_single_task(config, task_file_path)
            results[file_name] = result
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results[file_name] = {"success": False, "error": str(e)}
    
    # Save all results
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"all_tasks_{config.search_mode}_results.json")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"All results saved to {output_path}")
    
    # Print summary
    total = len(results)
    successful = sum(1 for r in results.values() if r.get("success", False))
    print(f"Overall results: {successful}/{total} tasks solved successfully ({successful/total*100:.2f}%)")
    
    return results


def show_help():
    """Show help information and parameter documentation"""
    print("\nrSTAR-ARC: Self-play muTuAl Reasoning for ARC\n")
    print("This program applies the rStar methodology to solve ARC (Abstraction and Reasoning Corpus) tasks.\n")
    
    print("Usage examples:")
    print("  Local run:  python main.py --task-index=1 --verbose")
    print("  Cluster run: ./run.sh --task=1 --gpus=1 --dtype=bfloat16 --verbose\n")
    
    print("Configuration:")
    print("  You can specify parameters via:")
    print("  1. Command line arguments")
    print("  2. Config file (--config-file=config/example_config.yaml)")
    print("  3. Default values from schema\n")
    
    # Show all available parameters
    CLI.print_available_params()
    
    print("\nFor more information, check the README.md file.")


if __name__ == '__main__':
    # Check for help flag as a special case
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
        
    args = CLI.parse_args()
    config = CLI.create_config(args)

    if config.all_tasks:
        run_all_tasks(config)
    else:
        run_single_task(config)

    print("Done!")

