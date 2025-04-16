import json
import os.path

import scipy.stats

from rstar_deepthink.arc_task import Grid
from constants import DATA_SAMPLE_DIR


def get_description(config, task_name: str) -> str:
    if config.evaluation:
        path = os.path.join(DATA_SAMPLE_DIR, "bootstrap", f"evaluation_descriptions_1.json")
    else:
        path = os.path.join(DATA_SAMPLE_DIR, "bootstrap", f"training_descriptions_4.json")

    key = f"{task_name}.json"

    # safely load the description from the json file
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            description = data.get(key, "Description not found.")
    except FileNotFoundError:
        description = "Description file not found."
    except json.JSONDecodeError:
        description = "Error decoding JSON file."

    return description


def normalized_similarity_score(correct_grids: list[Grid], predicted_grids: list[Grid]) -> float:
    """Calculate the normalized closeness between two lists of grids."""
    total_correct = 0
    total = 0

    for correct_grid, predicted_grid in zip(correct_grids, predicted_grids):
        if correct_grid.rows != predicted_grid.rows or correct_grid.columns != predicted_grid.columns:
            return -1.0

        closeness = 0
        for i in range(correct_grid.rows):
            for j in range(correct_grid.columns):
                if correct_grid.grid[i][j] == predicted_grid.grid[i][j]:
                    closeness += 1

        total_correct += closeness
        total += correct_grid.rows * correct_grid.columns

    if total_correct == total:
        return 1.0

    percentage_correct = total_correct / total

    # the best an incorrect solution can get is -0.5
    # this is to bias the model towards partially correct solutions, but not too much
    return -1 + percentage_correct / 2


def temperature_lerp(current_rollout, max_rollouts, min_temp, max_temp):
    """
    Calculates temperature using linear interpolation.

    Args:
      current_rollout: The index of the current rollout (0-based).
      max_rollouts: The total number of rollouts.
      min_temp: The minimum temperature.
      max_temp: The maximum temperature.

    Returns:
      The calculated temperature for the current rollout.
    """

    if max_rollouts == 1:
        return (min_temp + max_temp) / 2

    # TODO: fix formula to account for few calls to temperature changing function!

    return min_temp + (max_temp - min_temp) * (current_rollout / max_rollouts)


def temperature_beta_cdf(current_rollout, max_rollouts, min_temp, max_temp, target_fraction=0.5, concentration=0.75):
    """
    Calculates temperature using the Beta CDF for a mid-rollout slowdown.

    Args:
      current_rollout: The index of the current rollout (0-based).
      max_rollouts: The total number of rollouts.
      min_temp: The minimum temperature.
      max_temp: The maximum temperature.
      target_fraction: Fraction of rollouts (0 to 1, exclusive) around which
                       the temperature change rate is minimal.
      concentration: Controls the strength of the slowdown. Must be > 0.
                     Values < 1 cause slowdown (smaller is stronger dwell).
                     Values >= 1 cause S-curve (faster middle). Recommended: 0.05 to 0.5 for dwell.

    Returns:
      The calculated temperature for the current rollout.
    """

    # Calculate Beta distribution parameters
    a = target_fraction * concentration
    b = (1.0 - target_fraction) * concentration

    # Normalize rollout index to range [0, 1]
    # Handle edge case for x=0 or x=1 where CDF might be exactly 0 or 1
    if current_rollout <= 0:
        x = 0.0
    elif current_rollout >= max_rollouts - 1:
        x = 1.0
    else:
        x = current_rollout / (max_rollouts - 1)

    # Calculate the Beta CDF
    # scipy's beta.cdf handles x=0 and x=1 correctly
    f_x = scipy.stats.beta.cdf(x, a, b)

    # Scale output (0 to 1) to the temp range (min_temp to max_temp)
    return min_temp + (max_temp - min_temp) * f_x
