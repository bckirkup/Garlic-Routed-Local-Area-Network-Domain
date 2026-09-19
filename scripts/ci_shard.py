"""Print the test files belonging to one CI shard.

Test files are sorted and dealt round-robin across ``--total`` shards so that
every file lands in exactly one shard regardless of runner.

Usage: python scripts/ci_shard.py --index 1 --total 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"


def shard_files(index: int, total: int, tests_dir: Path = TESTS_DIR) -> list[Path]:
    if total < 1:
        raise ValueError("total must be >= 1")
    if not 1 <= index <= total:
        raise ValueError(f"index must be in 1..{total}, got {index}")
    files = sorted(tests_dir.glob("test_*.py"))
    return [path for position, path in enumerate(files) if position % total == index - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True, help="1-based shard index")
    parser.add_argument("--total", type=int, required=True, help="number of shards")
    args = parser.parse_args()
    for path in shard_files(args.index, args.total):
        print(path.relative_to(TESTS_DIR.parent).as_posix())


if __name__ == "__main__":
    main()
