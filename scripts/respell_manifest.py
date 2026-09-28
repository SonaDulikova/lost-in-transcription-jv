"""Write a copy of a manifest with its text respelled to convo 5's conventions (lit/respell.py).

    uv run scripts/respell_manifest.py data/central_jv/train.csv --out data/central_jv/train_c5spell.csv
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from lit.respell import respell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, keep_default_na=False)
    old, new = df["text"], df["text"].map(respell)
    changed = Counter()
    for old_t, new_t in zip(old, new):
        changed.update(f"{a}->{b}" for a, b in zip(old_t.split(), new_t.split()) if a != b)
    rows = int((new != old).sum())
    df["text"] = new
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {rows} of {len(df)} rows changed, {sum(changed.values())} words")
    for pair, n in changed.most_common(15):
        print(f"  {n:5} {pair}")


if __name__ == "__main__":
    main()
