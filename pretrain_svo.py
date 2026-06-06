"""Compatibility entrypoint for src.scripts.pretrain_svo."""

from runpy import run_module


if __name__ == "__main__":
    run_module("src.scripts.pretrain_svo", run_name="__main__")

