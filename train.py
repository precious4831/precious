"""Compatibility entrypoint for src.scripts.train."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.scripts.train", run_name="__main__")

