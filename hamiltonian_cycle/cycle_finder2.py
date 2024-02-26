import random
import math

# source: ChatGPT 3.5 and https://clisby.net/projects/hamiltonian_path/

# faster version

def in_sublattice(x, y, xmax, ymax):
    if x < 0 or x > xmax or y < 0 or y > ymax:
        return False
    return True

def reverse_path(i1, i2, path):
    jlim = (i2 - i1 + 1) // 2
    for j in range(jlim):
        path[i1+j], path[i2-j] = path[i2-j], path[i1+j]

def backbite_left(step, n, path, xmax, ymax):
    neighbour = (path[0][0] + step[0], path[0][1] + step[1])
    if in_sublattice(neighbour[0], neighbour[1], xmax, ymax):
        in_path = False
        for j in range(1, n, 2):
            if neighbour == path[j]:
                in_path = True
                break
        if in_path:
            reverse_path(0, j-1, path)
        else:
            reverse_path(0, n-1, path)
            path.append(neighbour)
            n += 1
    return n

def backbite_right(step, n, path, xmax, ymax):
    neighbour = (path[n-1][0] + step[0], path[n-1][1] + step[1])
    if in_sublattice(neighbour[0], neighbour[1], xmax, ymax):
        in_path = False
        for j in range(n-2, -1, -2):
            if neighbour == path[j]:
                in_path = True
                break
        if in_path:
            reverse_path(j+1, n-1, path)
        else:
            path.append(neighbour)
            n += 1
    return n

def backbite(n, path, xmax, ymax):
    step = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    if random.randint(0, 1) == 0:
        n = backbite_left(step, n, path, xmax, ymax)
    else:
        n = backbite_right(step, n, path, xmax, ymax)
    return n

def generate_hamiltonian_path(q, xmax, ymax, must_fill=False):
    path = [(random.randint(0, xmax), random.randint(0, ymax))]
    n = 1
    if must_fill:
        nattempts = 1 + q * 10.0 * (xmax + 1) * (ymax + 1) * math.pow(math.log(2. + (xmax + 1) * (ymax + 1)), 2)
        while n < (xmax + 1) * (ymax + 1):
            for _ in range(int(nattempts)):
                n = backbite(n, path, xmax, ymax)
    else:
        nattempts = q * 10.0 * (xmax + 1) * (ymax + 1) * math.pow(math.log(2. + (xmax + 1) * (ymax + 1)), 2)
        for _ in range(int(nattempts)):
            n = backbite(n, path, xmax, ymax)
    return n, path

def generate_hamiltonian_circuit(q, xmax, ymax):
    n, path = generate_hamiltonian_path(q, xmax, ymax)
    #nmax = xmax * ymax
    min_dist = 1 + (n % 2)
    while abs(path[n-1][0] - path[0][0]) + abs(path[n-1][1] - path[0][1]) != min_dist:
        n = backbite(n, path, xmax, ymax)
    return n, path