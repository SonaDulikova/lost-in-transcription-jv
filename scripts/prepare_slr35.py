"""Build data/slr35/mapping.tsv (file, text) from the OpenSLR SLR35 index and the shards on disk.

Then pack the audio with scripts/prepare_tts.py:
  uv run scripts/prepare_tts.py --tsv data/slr35/mapping.tsv --audio-dir data/slr35/asr_javanese/data \
      --file-col file --text-col text --ext .flac --session slr35 --out data/slr35
"""

import argparse
import sys
from pathlib import Path

from lit.data import REPO, load_slr35_index, slr35_mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=REPO / "data" / "slr35" / "utt_spk_text.tsv")
    ap.add_argument("--audio-dir", type=Path, default=REPO / "data" / "slr35" / "asr_javanese" / "data")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "slr35" / "mapping.tsv")
    args = ap.parse_args()

    index = load_slr35_index(args.index)
    available = {p.stem for p in args.audio_dir.rglob("*.flac")}
    mapping = slr35_mapping(index, available)
    if mapping.empty:
        print(f"ERROR: no index rows have audio under {args.audio_dir}", file=sys.stderr)
        sys.exit(1)
    speakers = index[index["id"].isin(available)]["speaker"].nunique()
    print(f"index rows {len(index)}, flac files {len(available)}, mapped {len(mapping)}, "
          f"flac without index text {len(available) - len(mapping)}, speakers {speakers}")
    mapping.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
