import numpy as np
import random

#source: ChatGPT 3.5

#VERY SLOW

def grid_generator(width, height):
    grid = []
    for i in range(height):
        grid.append(list(np.arange(width) + 1 + i*width))
    return grid

def hamiltonian_cycle_from_point(grid, start_row, start_col): 
    def is_valid_move(grid, path, row, col):
        # Check if the move is within the grid and the cell is not already visited
        if (row >= 0 and row < len(grid[0])) and (col >= 0 and col < len(grid)):
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
    raise Exception("Oopsie the rows and columns are both odd or smth idk but it no work sad fact :(")