"""Compatibility entrypoint for src.scripts.data_collector."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.scripts.data_collector", run_name="__main__")

