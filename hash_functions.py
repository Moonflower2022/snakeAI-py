def hash_state(snake, fruit):
    def hash_2d(array_2d):
        return tuple(tuple(inner_array) for inner_array in array_2d)
    return (hash_2d(snake), tuple(fruit))