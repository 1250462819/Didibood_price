#!/usr/bin/env python3
"""Create price service tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.session import init_db


def main() -> None:
    init_db()
    print("Price database tables are ready.")


if __name__ == "__main__":
    main()
