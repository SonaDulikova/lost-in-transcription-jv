"""Create data/indonesian_dev/ground_truth.csv and check it against the runtime's dev copy."""

from pathlib import Path

import pandas as pd

from lit.data import DEV_DIR, load_dev

RUNTIME_DATA = Path.home() / "repos" / "lost-in-transcription-runtime" / "data" / "data"


def main() -> None:
    df = load_dev(DEV_DIR)
    out = DEV_DIR / "ground_truth.csv"
    df[["audio_filename", "transcript"]].to_csv(out, index=False)
    print(f"wrote {out} with {len(df)} rows")

    fmt = RUNTIME_DATA / "submission_format.csv"
    if fmt.exists():
        f = pd.read_csv(fmt)
        same = list(f["audio_filename"]) == list(df["audio_filename"])
        print(f"runtime submission_format.csv order matches metadata.tsv: {same}")
        assert set(f["audio_filename"]) == set(df["audio_filename"]), "filename sets differ"
    else:
        print(f"runtime dev copy not found at {fmt}; skipped cross-check")


if __name__ == "__main__":
    main()
