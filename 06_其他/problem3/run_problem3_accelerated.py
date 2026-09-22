#!/usr/bin/env python3
"""CLI wrapper that swaps only the Q3 hot loop for the accelerated kernel."""

from problem3 import solve_problem3
from problem3.accelerated_problem3 import solve_model_accelerated


solve_problem3.solve_model = solve_model_accelerated


if __name__ == "__main__":
    raise SystemExit(solve_problem3.main())
