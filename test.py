"""Compatibility entrypoint for src.scripts.test."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.scripts.test", run_name="__main__")

