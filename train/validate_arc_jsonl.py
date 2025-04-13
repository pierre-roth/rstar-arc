import argparse
import json
import os
import sys
from typing import List, Any

# Define expected top-level keys and their types
EXPECTED_TOP_LEVEL_KEYS = {
    "task_name": str,
    "task_json": dict,
    "solution": str,
    "weight": float,
}

# Define expected structure within task_json
EXPECTED_TASK_JSON_KEYS = {
    "train": list,
    "test": list,
}

# Define expected structure for items within train/test lists
EXPECTED_IO_PAIR_KEYS = {
    "input": list,
    "output": list,
}


def validate_grid(grid: Any, context: str) -> List[str]:
    """Validates if an object is a list of lists of integers."""
    errors = []
    if not isinstance(grid, list):
        errors.append(f"{context}: Expected a list (outer list), got {type(grid).__name__}.")
        return errors  # Stop further checks if not a list

    if not grid:
        # Allow empty grids? Or require at least one row? Assuming allowed for now.
        # errors.append(f"{context}: Grid list is empty.")
        pass  # Allow empty grids

    for i, row in enumerate(grid):
        if not isinstance(row, list):
            errors.append(f"{context}: Row {i} is not a list, got {type(row).__name__}.")
            continue  # Skip checking elements if row is not a list
        if not row:
            # Allow empty rows? Assuming allowed for now.
            # errors.append(f"{context}: Row {i} is an empty list.")
            pass  # Allow empty rows

        for j, cell in enumerate(row):
            if not isinstance(cell, int):
                errors.append(f"{context}: Cell ({i},{j}) is not an integer, got {type(cell).__name__} ('{cell}').")
    return errors


def validate_io_pair(pair: Any, context: str) -> List[str]:
    """Validates the structure of an input/output pair dictionary."""
    errors = []
    if not isinstance(pair, dict):
        errors.append(f"{context}: Expected a dictionary, got {type(pair).__name__}.")
        return errors

    # Check required keys
    missing_keys = [k for k in EXPECTED_IO_PAIR_KEYS if k not in pair]
    if missing_keys:
        errors.append(f"{context}: Missing required keys: {', '.join(missing_keys)}.")

    extra_keys = [k for k in pair if k not in EXPECTED_IO_PAIR_KEYS]
    if extra_keys:
        errors.append(f"{context}: Found unexpected keys: {', '.join(extra_keys)}.")

    # Validate 'input' grid if present
    if "input" in pair:
        errors.extend(validate_grid(pair["input"], f"{context}['input']"))
    # Validate 'output' grid if present
    if "output" in pair:
        errors.extend(validate_grid(pair["output"], f"{context}['output']"))

    return errors


def validate_task_json(task_data: Any, context: str) -> List[str]:
    """Validates the structure of the task_json dictionary."""
    errors = []
    if not isinstance(task_data, dict):
        errors.append(f"{context}: Expected a dictionary, got {type(task_data).__name__}.")
        return errors

    # Check required keys ('train', 'test')
    missing_keys = [k for k in EXPECTED_TASK_JSON_KEYS if k not in task_data]
    if missing_keys:
        errors.append(f"{context}: Missing required keys: {', '.join(missing_keys)}.")

    extra_keys = [k for k in task_data if k not in EXPECTED_TASK_JSON_KEYS]
    if extra_keys:
        errors.append(f"{context}: Found unexpected keys: {', '.join(extra_keys)}.")

    # Validate 'train' list if present
    if "train" in task_data:
        train_list = task_data["train"]
        if not isinstance(train_list, list):
            errors.append(f"{context}['train']: Expected a list, got {type(train_list).__name__}.")
        else:
            if not train_list:
                errors.append(f"{context}['train']: List is empty (at least one training example is usually expected).")
            for i, pair in enumerate(train_list):
                errors.extend(validate_io_pair(pair, f"{context}['train'][{i}]"))

    # Validate 'test' list if present
    if "test" in task_data:
        test_list = task_data["test"]
        if not isinstance(test_list, list):
            errors.append(f"{context}['test']: Expected a list, got {type(test_list).__name__}.")
        else:
            if not test_list:
                errors.append(f"{context}['test']: List is empty (at least one test example is usually expected).")
            for i, pair in enumerate(test_list):
                errors.extend(validate_io_pair(pair, f"{context}['test'][{i}]"))

    return errors


def validate_jsonl_file(file_path: str) -> bool:
    """
    Validates a JSON Lines file against the expected ARC task structure.

    Args:
        file_path: Path to the .jsonl file.

    Returns:
        True if the file is valid, False otherwise.
    """
    print(f"\n--- Validating file: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False

    is_valid = True
    total_lines = 0
    error_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                line_errors: List[str] = []
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        line_errors.append(f"Expected JSON object (dict), got {type(data).__name__}.")
                        data = {}  # Set to empty dict to allow key checks below

                    # 1. Check top-level keys and types
                    missing_keys = [k for k in EXPECTED_TOP_LEVEL_KEYS if k not in data]
                    if missing_keys:
                        line_errors.append(f"Missing top-level keys: {', '.join(missing_keys)}.")

                    extra_keys = [k for k in data if k not in EXPECTED_TOP_LEVEL_KEYS]
                    if extra_keys:
                        line_errors.append(f"Found unexpected top-level keys: {', '.join(extra_keys)}.")

                    for key, expected_type in EXPECTED_TOP_LEVEL_KEYS.items():
                        if key in data:
                            # Special check for float allows int as well, but weight should ideally be float
                            if expected_type == float and not isinstance(data[key], (float, int)):
                                line_errors.append(
                                    f"Key '{key}': Expected float, got {type(data[key]).__name__} ('{data[key]}').")
                            elif expected_type != float and not isinstance(data[key], expected_type):
                                line_errors.append(
                                    f"Key '{key}': Expected {expected_type.__name__}, got {type(data[key]).__name__}.")

                    # 2. Validate 'task_json' structure if it's a dict
                    if "task_json" in data and isinstance(data["task_json"], dict):
                        line_errors.extend(validate_task_json(data["task_json"], "task_json"))
                    elif "task_json" in data:
                        # Error already added above if type is wrong
                        pass


                except json.JSONDecodeError as e:
                    line_errors.append(f"Invalid JSON: {e}")
                except Exception as e:
                    # Catch unexpected errors during validation
                    line_errors.append(f"Unexpected validation error: {e}")

                if line_errors:
                    is_valid = False
                    error_count += 1
                    print(f"Error(s) on line {line_num}:")
                    for err in line_errors:
                        print(f"  - {err}")
                    # Optionally print problematic line content (truncated)
                    # print(f"    Content: {line[:150]}{'...' if len(line) > 150 else ''}")


    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return False

    print(f"--- Validation Summary for {file_path} ---")
    print(f"Total lines processed: {total_lines}")
    print(f"Lines with errors: {error_count}")
    if is_valid:
        print("Result: File appears valid.")
    else:
        print("Result: File has validation errors.")
    print("-" * (len(file_path) + 28))  # Adjust separator length
    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Validate ARC task JSON Lines (.jsonl) files.")
    parser.add_argument("files", nargs='+', help="Path(s) to the .jsonl file(s) to validate.")

    args = parser.parse_args()

    overall_success = True
    for file_path in args.files:
        if not validate_jsonl_file(file_path):
            overall_success = False

    if not overall_success:
        print("\nOne or more files failed validation.")
        sys.exit(1)  # Exit with error code if any file fails
    else:
        print("\nAll specified files passed validation.")
        sys.exit(0)  # Exit with success code


if __name__ == "__main__":
    main()
