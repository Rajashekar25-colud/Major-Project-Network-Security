import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_csvs(folder: str | Path, pattern: str = "*.csv") -> list[Path]:
    p = Path(folder)
    if p.is_file():
        return [p]
    files = sorted(x for x in p.rglob(pattern) if x.is_file() and "states" not in x.name.lower())
    if not files:
        raise SystemExit(f"No CSV files found in {p.resolve()}")
    return files
