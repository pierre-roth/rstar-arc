import json
import argparse
import os
import sys
from typing import List, Dict, Any

# Define expected top-level keys and their types
EXPECTED_TOP_LEVEL_KEYS = {
    "task_name": str,
    "task_json": dict,
    "solution": str,
    "weight": float,
}

# Define expected structure within task_json
# We know 'name' might appear, but we *don't* list it here
# as we want the script to flag/remove unexpected keys.
EXPECTED_TASK_JSON_KEYS = {
    "train": list,
    "test": list,
}

# Define expected structure for items within train/test lists
EXPECTED_IO_PAIR_KEYS = {
    "input": list,
    "output": list,
}

# --- Validation Helper Functions (largely unchanged) ---

def validate_grid(grid: Any, context: str) -> List[str]:
    """Validates if an object is a list of lists of integers."""
    errors = []
    if not isinstance(grid, list):
        errors.append(f"{context}: Expected a list (outer list), got {type(grid).__name__}.")
        return errors

    if not grid:
         pass # Allow empty grids

    for i, row in enumerate(grid):
        if not isinstance(row, list):
            errors.append(f"{context}: Row {i} is not a list, got {type(row).__name__}.")
            continue
        if not row:
            pass # Allow empty rows

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

    missing_keys = [k for k in EXPECTED_IO_PAIR_KEYS if k not in pair]
    if missing_keys:
        errors.append(f"{context}: Missing required keys: {', '.join(missing_keys)}.")

    extra_keys = [k for k in pair if k not in EXPECTED_IO_PAIR_KEYS]
    if extra_keys:
         errors.append(f"{context}: Found unexpected keys: {', '.join(extra_keys)}.")

    if "input" in pair:
        errors.extend(validate_grid(pair["input"], f"{context}['input']"))
    if "output" in pair:
        errors.extend(validate_grid(pair["output"], f"{context}['output']"))

    return errors

def validate_task_json(task_data: Any, context: str) -> List[str]:
    """Validates the structure of the task_json dictionary."""
    errors = []
    if not isinstance(task_data, dict):
        errors.append(f"{context}: Expected a dictionary, got {type(task_data).__name__}.")
        return errors

    missing_keys = [k for k in EXPECTED_TASK_JSON_KEYS if k not in task_data]
    if missing_keys:
        errors.append(f"{context}: Missing required keys: {', '.join(missing_keys)}.")

    # Check for extra keys *before* potentially removing 'name'
    # We still want to report other unexpected keys if they exist.
    extra_keys = [k for k in task_data if k not in EXPECTED_TASK_JSON_KEYS and k != 'name'] # Exclude 'name' for now
    if extra_keys:
         errors.append(f"{context}: Found unexpected keys: {', '.join(extra_keys)}.")

    if "train" in task_data:
        train_list = task_data["train"]
        if not isinstance(train_list, list):
            errors.append(f"{context}['train']: Expected a list, got {type(train_list).__name__}.")
        else:
            # Allow empty train list? Usually not desired, but maybe valid in some cases.
            # if not train_list:
            #      errors.append(f"{context}['train']: List is empty.")
            for i, pair in enumerate(train_list):
                errors.extend(validate_io_pair(pair, f"{context}['train'][{i}]"))

    if "test" in task_data:
        test_list = task_data["test"]
        if not isinstance(test_list, list):
            errors.append(f"{context}['test']: Expected a list, got {type(test_list).__name__}.")
        else:
            # Allow empty test list?
            # if not test_list:
            #      errors.append(f"{context}['test']: List is empty.")
            for i, pair in enumerate(test_list):
                errors.extend(validate_io_pair(pair, f"{context}['test'][{i}]"))

    return errors

# --- Main Validation and Cleaning Function ---

def validate_and_clean_jsonl_file(input_file_path: str, output_file_path: str) -> Dict[str, Any]:
    """
    Validates and cleans a JSON Lines file, removing the 'name' key from 'task_json'.

    Args:
        input_file_path: Path to the input .jsonl file.
        output_file_path: Path to the output .jsonl file to write cleaned data.

    Returns:
        A dictionary containing validation results:
        {
            "total_lines": int,
            "error_lines": int,
            "fixed_lines": int, # Lines where 'name' key was removed
            "has_errors": bool # True if unfixable errors remain
        }
    """
    results = {
        "total_lines": 0,
        "error_lines": 0,
        "fixed_lines": 0,
        "has_errors": False
    }
    print(f"\n--- Processing file: {input_file_path} ---")
    print(f"--- Writing cleaned output to: {output_file_path} ---")

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found: {input_file_path}")
        results["has_errors"] = True
        return results # Cannot proceed

    try:
        # Ensure output directory exists if needed
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                results["total_lines"] += 1
                line = line.strip()
                if not line: # Skip empty lines, don't write to output
                    continue

                line_errors: List[str] = []
                line_fixed = False
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                         line_errors.append(f"Expected JSON object (dict), got {type(data).__name__}.")
                         data = {} # Allow checks below

                    # 1. Check top-level keys and types
                    missing_keys = [k for k in EXPECTED_TOP_LEVEL_KEYS if k not in data]
                    if missing_keys:
                        line_errors.append(f"Missing top-level keys: {', '.join(missing_keys)}.")

                    # Report extra keys at top level, but don't remove them automatically
                    extra_keys = [k for k in data if k not in EXPECTED_TOP_LEVEL_KEYS]
                    if extra_keys:
                         line_errors.append(f"Found unexpected top-level keys: {', '.join(extra_keys)}.")

                    for key, expected_type in EXPECTED_TOP_LEVEL_KEYS.items():
                        if key in data:
                            if expected_type == float and not isinstance(data[key], (float, int)):
                                line_errors.append(f"Key '{key}': Expected float, got {type(data[key]).__name__} ('{data[key]}').")
                            elif expected_type != float and not isinstance(data[key], expected_type):
                                line_errors.append(f"Key '{key}': Expected {expected_type.__name__}, got {type(data[key]).__name__}.")

                    # 2. Validate 'task_json' structure and CLEAN 'name' key
                    if "task_json" in data and isinstance(data["task_json"], dict):
                        # *** CLEANING STEP ***
                        if 'name' in data["task_json"]:
                            del data["task_json"]['name']
                            line_fixed = True
                            results["fixed_lines"] += 1
                            # Optionally log the fix: print(f"Note on line {line_num}: Removed 'name' key from 'task_json'.")

                        # Validate the rest of the task_json structure
                        line_errors.extend(validate_task_json(data["task_json"], "task_json"))
                    elif "task_json" in data:
                        # Type error already added above
                        pass

                except json.JSONDecodeError as e:
                    line_errors.append(f"Invalid JSON: {e}")
                    # Cannot write invalid JSON, skip writing this line to output
                    print(f"Error(s) on line {line_num}:")
                    print(f"  - {line_errors[0]}") # Print JSON error
                    print(f"    Content: {line[:150]}{'...' if len(line) > 150 else ''}")
                    results["error_lines"] += 1
                    results["has_errors"] = True
                    continue # Skip writing this line
                except Exception as e:
                    line_errors.append(f"Unexpected validation error: {e}")

                # Report remaining errors (after potential fix)
                if line_errors:
                    results["error_lines"] += 1
                    results["has_errors"] = True
                    print(f"Error(s) on line {line_num} (after potential fix):")
                    for err in line_errors:
                        print(f"  - {err}")

                # Write the (potentially cleaned) data to the output file
                try:
                    outfile.write(json.dumps(data) + '\n')
                except Exception as write_e:
                    print(f"Error writing processed data for line {line_num} to {output_file_path}: {write_e}")
                    results["has_errors"] = True
                    # Depending on severity, maybe stop processing the file


    except IOError as e:
        print(f"Error reading input file {input_file_path} or writing output file {output_file_path}: {e}")
        results["has_errors"] = True
    except Exception as e:
         print(f"An unexpected error occurred while processing {input_file_path}: {e}")
         results["has_errors"] = True

    print(f"--- Processing Summary for {input_file_path} ---")
    print(f"Total lines processed: {results['total_lines']}")
    print(f"Lines fixed (removed 'task_json.name'): {results['fixed_lines']}")
    print(f"Lines with remaining errors: {results['error_lines']}")
    if not results["has_errors"] and results["fixed_lines"] > 0:
        print(f"Result: File cleaned successfully and saved to {output_file_path}.")
    elif not results["has_errors"] and results["fixed_lines"] == 0:
         print(f"Result: File was already valid. Output written to {output_file_path}.")
    else:
        print(f"Result: File has unrecoverable errors or failed during processing. Check messages above.")
        if os.path.exists(output_file_path):
             print(f"Note: Output file '{output_file_path}' may be incomplete or contain errors.")
    print("-" * (len(input_file_path) + 28)) # Adjust separator length
    return results

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean ARC task JSON Lines (.jsonl) files by removing the 'name' key from 'task_json'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument("files", nargs='+', help="Path(s) to the input .jsonl file(s).")
    parser.add_argument(
        "--output-dir",
        help="Directory to save cleaned files. If not specified, cleaned files are saved next to originals."
    )
    parser.add_argument(
        "--suffix",
        default="_cleaned",
        help="Suffix to append to original filenames when saving cleaned files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input files directly instead of creating new files. USE WITH CAUTION!"
    )

    args = parser.parse_args()

    overall_success = True
    any_fixes_made = False

    for input_path in args.files:
        if not os.path.isfile(input_path):
             print(f"Warning: Input path is not a file, skipping: {input_path}")
             continue

        output_path = ""
        if args.overwrite:
            output_path = input_path # Output is same as input
            print(f"Warning: Overwriting input file: {input_path}")
            # Consider adding a confirmation step here in a real application
        elif args.output_dir:
            base_name = os.path.basename(input_path)
            # Optionally add suffix even if output dir is specified
            # name, ext = os.path.splitext(base_name)
            # output_base_name = f"{name}{args.suffix}{ext}"
            output_path = os.path.join(args.output_dir, base_name)
        else:
            # Save next to original with suffix
            dir_name = os.path.dirname(input_path)
            base_name = os.path.basename(input_path)
            name, ext = os.path.splitext(base_name)
            if not ext.lower() == ".jsonl":
                 print(f"Warning: Input file '{input_path}' does not have .jsonl extension. Output suffix might look odd.")
                 ext = ".jsonl" # Force .jsonl extension for output? Or keep original? Keeping original for now.

            output_base_name = f"{name}{args.suffix}{ext}"
            output_path = os.path.join(dir_name, output_base_name)

        # Prevent accidental overwrite if paths somehow end up the same without --overwrite
        if not args.overwrite and os.path.abspath(input_path) == os.path.abspath(output_path):
             print(f"Error: Input and output paths are identical ('{input_path}'). Use --overwrite or change --output-dir/--suffix.")
             overall_success = False
             continue


        results = validate_and_clean_jsonl_file(input_path, output_path)

        if results["has_errors"]:
            overall_success = False
        if results["fixed_lines"] > 0:
             any_fixes_made = True


    print("\n--- Overall Summary ---")
    if not overall_success:
        print("One or more files had unrecoverable errors during processing.")
        sys.exit(1) # Exit with error code
    elif any_fixes_made:
        print("Processing complete. One or more files were cleaned.")
        sys.exit(0) # Exit with success code
    else:
         print("Processing complete. All specified files were already valid (or no fixes were needed).")
         sys.exit(0) # Exit with success code


if __name__ == "__main__":
    main()
