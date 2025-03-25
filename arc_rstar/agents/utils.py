from __future__ import annotations


def normalized_similarity(correct_grids: list["Grid"], predicted_grids: list["Grid"]) -> float:
    """Calculate the normalized closeness between two lists of grids."""
    total = 0
    for correct_grid, predicted_grid in zip(correct_grids, predicted_grids):
        if correct_grid.rows != predicted_grid.rows or correct_grid.columns != predicted_grid.columns:
            return -1.0

        closeness = 0
        for i in range(correct_grid.rows):
            for j in range(correct_grid.columns):
                if correct_grid.grid[i][j] == predicted_grid.grid[i][j]:
                    closeness += 1

        total += closeness / (correct_grid.rows * correct_grid.columns)

    return total / len(correct_grids)
