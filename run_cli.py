from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))

    from multi_agent_research_lab.cli import app

    app()


if __name__ == "__main__":
    main()

