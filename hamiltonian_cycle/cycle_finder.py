import numpy as np
import random

def hamiltonian_cycle_from_point(grid, start_row, start_col):
    def is_valid_move(grid, path, row, col):
        # Check if the move is within the grid and the cell is not already visited
        if (row >= 0 and row < len(grid)) and (col >= 0 and col < len(grid[0])):
            if (row, col) not in path:
                return True
        return False
    def dfs(grid, path, row, col):
        path.append((row, col))

        if len(path) == len(grid) * len(grid[0]):
            # Check if the last cell is adjacent to the start cell
            last_row, last_col = path[-1]
            if abs(last_row - start_row) + abs(last_col - start_col) == 1:
                return True
        
        # Possible moves
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(moves)

        for move in moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if is_valid_move(grid, path, new_row, new_col):
                if dfs(grid, path, new_row, new_col):
                    return True
        
        path.pop() # Backtrack
        return False

    # Start from the specified point and try to find a Hamiltonian cycle
    path = []
    if dfs(grid, path, start_row, start_col):
        return path
    return None

def grid_generator(width, height):
    grid = []
    for i in range(height):
        grid.append(list(np.arange(width) + 1 + i*width))
    return grid

def cycle_to_directions(cycle):
    directions = {}
    for i in range(len(cycle)):
        current_point = cycle[i]
        next_point = cycle[(i + 1) % len(cycle)]  # Wrap around to the first point if we're at the last point
        direction = (next_point[0] - current_point[0], next_point[1] - current_point[1])
        directions[current_point] = direction
    return directions

# Example usage:

grid = grid_generator(7, 6)

print(grid)

y, x = 0, 1  # Start from the center (5)

cycle = hamiltonian_cycle_from_point(grid, y, x)
if cycle:
    print("Hamiltonian cycle found:")
    print(cycle)
    print(cycle_to_directions(cycle))
else:
    print("No Hamiltonian cycle found")
