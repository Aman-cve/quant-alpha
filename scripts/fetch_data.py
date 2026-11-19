"""Refresh the local CSV cache for the full US + India universe.

    python3 scripts/fetch_data.py          # fill cache (skip symbols already cached)
    python3 scripts/fetch_data.py --refresh  # force re-download everything
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantalpha import data, universe  # noqa: E402


def main() -> None:
    refresh = "--refresh" in sys.argv
    syms = universe.all_symbols()
    ok, fail = [], []
    for s in syms:
        try:
            ser = data.load(s, rng="2y", refresh=refresh)
            print(f"  {s:14} {len(ser):4d} days  (last {ser.dates[-1]}, {ser.close[-1]:.2f})")
            ok.append(s)
        except Exception as e:  # noqa: BLE001
            print(f"  {s:14} FAILED: {e}")
            fail.append(s)
    print(f"\ncached {len(ok)}/{len(syms)} symbols" + (f", failed: {fail}" if fail else ""))
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
