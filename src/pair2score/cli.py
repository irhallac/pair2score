from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Pair2Score compact runner")
    parser.add_argument("config", nargs="?", default="configs/pipeline.yaml", help="Pipeline config path")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
