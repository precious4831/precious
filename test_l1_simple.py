"""Compatibility entrypoint for src.scripts.test_l1_simple."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.scripts.test_l1_simple", run_name="__main__")

