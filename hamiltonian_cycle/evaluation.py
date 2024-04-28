from testing import generate_random_snake, generate_hamiltonian_circuit, cycle_to_directions, directions_move
from game import Game

trials = 5000

snake_length = 1
width = 4
height = 4

wins = 0
total_win_steps = 0

print("Starving evaluation...")

for i in range(trials):
    print(f"Trials: {i}/{trials}", end="\r")

    snake = generate_random_snake(snake_length, width, height)

    #grid = grid_generator(width, height)
    #cycle = hamiltonian_cycle_from_point(grid, snake[0][1], snake[0][0])

    q = 1
    n, cycle = generate_hamiltonian_circuit(q, width - 1, height - 1)

    directions = cycle_to_directions(cycle)

    game = Game(directions_move, width, height, snake, cycle, directions)

    while not game.game_over:
        game.update()

    if game.won:
        wins += 1
        total_win_steps += game.steps

print("win ratio:", wins/trials)
if wins == 0:
    print("no wins")
else:
    print("average steps to win:", total_win_steps/wins)
    


