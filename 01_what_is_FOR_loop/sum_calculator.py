"""
1.	What is FOR loop?

You have a positive integer number N as an input. Please write a program in Python 3 
that calculates the sum in range 1 and N.

Limitations:
N <= 10^25;
Execution time: 0.1 seconds.

"""

import time
import decimal

# import numpy as np


# def sum_range(n):
#     return sum(range(1, n + 1))


# def sum_range_numpy(n):
#     return np.sum(np.arange(1, n + 1))


def math_sum(n):
    return n * (n + 1) // 2


def main():
    while True:
        print(
            "(q-Exit) Please enter N as positive integer or using scientific notation (1e1)."
        )

        input_val = input()

        if input_val == "q":
            break

        try:  # Tring to convert input value to integer
            n = int(input_val)
        except ValueError:
            try:  # If the input value cannot be converted to an integer, check whether it is represented in scientific notation
                base, power = input_val.lower().split("e")
                n = int(
                    decimal.Decimal(base) * 10 ** int(power)
                )  # Using Decimal instead of float to maintain accuracy for large numbers
            except (decimal.InvalidOperation, ValueError):
                print("N should be integer or number in scientific notation")
                continue

        if n <= 0:
            print("N should be positive")
            continue

        # print(n)

        start = time.perf_counter()
        n_sum = math_sum(n)
        end = time.perf_counter() - start

        print(f"Sum in range 1 to N = {n_sum}, elapsed: {end:0.8f}s\n")


if __name__ == "__main__":
    main()
