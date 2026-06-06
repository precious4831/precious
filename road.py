"""Compatibility entrypoint for src.scripts.road."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.scripts.road", run_name="__main__")

