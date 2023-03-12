"""
2.	Counting islands

You have a matrix MxN that represents a map. There are 2 possible states on the map: 1 - islands, 
0 - ocean. Your task is to calculate the number of islands in the most effective way. 
Please write code in Python 3.

Inputs:
M N
Matrix
"""

import random
import time

# from itertools import accumulate


def gen_map(length):
    """Generator of a given length sequence of random binary values."""
    for _ in range(length):
        yield random.randint(0, 1)


def count_islands(islands_map):
    """Calculator for the sum of positive occurrences in an array"""
    # return max(accumulate(islands_map))
    # return islands_map.count(1)
    return sum(islands_map)


def main():
    while True:
        print("(q-Exit) Enter size of the map:\n(M N)")

        map_size = input()  # Line input reader for map size

        if map_size == "q":  # Stop script if input is equal to 'q'
            break

        try:  # Attempting to read map size from input string
            rows, cols = map_size.split(" ")  # Line should be splitable by space
            rows, cols = int(rows), int(cols)  # and contain two integers
        except (ValueError, AttributeError):
            print(
                "Size of the map should be entered as 2 integers separated by 'space'\n"
            )
            continue

        if rows <= 0 or cols <= 0:  # The map size cannot be less than zero
            print("M N should be positive values")
            continue

        print(
            "\n(q-Exit) Enter 1 to generate random map, 'Return' to enter the map manually"
        )

        map_type = input()  # Line input reader for map way of generation
        islands_map = []
        row = 0

        if map_type == "1":  # Auto generating map if input is equal to '1'
            islands_map = gen_map(rows * cols)
        elif map_type == "q":
            break
        elif map_type == "":
            print(
                f"\n(q-Exit) Enter map (zeros and ones) of size ({rows}, {cols}):\n(press 'Return' after each row)"
            )
            while True:
                row_input = input()  # Line input reader for rows of a map

                if row_input == "q":
                    break

                try:  # Attempting to break line input into separate values
                    map_row = row_input.split(" ")
                except AttributeError:
                    print(
                        "Row input should contain integer numbers separated by 'space'"
                    )
                    continue

                if set(map_row).difference(
                    {"0", "1"}
                ):  # Checking if input values are binary
                    print("Row values should be zeros or ones\n")
                    continue

                if (
                    len(map_row) != cols
                ):  # Entered line length must be equal to the previously specified 'N' value
                    print(f"Please enter the number of values equal to {cols}\n")
                    continue

                islands_map.extend(
                    [int(i) for i in map_row]
                )  # Save input to list. There is no need to preserve dimensionality.

                row += 1  # Counting the number of already entered map rows
                if row == rows:
                    break
                # islands_map = [int(i) for row in islands_map for i in row]

        else:
            continue

        # print(islands_map)

        start = time.perf_counter()
        islands_num = count_islands(islands_map)  # Counting number of islands
        end = time.perf_counter() - start

        print(f"\nNumber of islands = {islands_num}, elapsed: {end:0.8f}s\n")


# islands_num = [[random.randint(0, 1) for _ in range(cols)] for _ in range(cols)]
# islands_num = np.array([gen_2d_map(rows, cols)])


if __name__ == "__main__":
    main()
