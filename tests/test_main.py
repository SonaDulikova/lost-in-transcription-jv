import importlib.util
import sys
from pathlib import Path

import pandas as pd

MAIN = Path(__file__).resolve().parents[1] / "submission_src" / "main.py"


def _load_main():
    spec = importlib.util.spec_from_file_location("sub_main", MAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sub_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_writes_submission_in_format_order(tmp_path: Path):
    m = _load_main()
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    for n in ["b.mp3", "a.mp3"]:
        (data / "clips" / n).write_bytes(b"")
    (data / "submission_format.csv").write_text("audio_filename,transcript\nb.mp3,\na.mp3,\n")
    out = tmp_path / "submission" / "submission.csv"

    n = m.run(lambda p: f"text for {p.name}", data, out)

    df = pd.read_csv(out, keep_default_na=False)
    assert n == 2
    assert list(df.columns) == ["audio_filename", "transcript"]
    assert df["audio_filename"].tolist() == ["b.mp3", "a.mp3"]
    assert df["transcript"].tolist() == ["text for b.mp3", "text for a.mp3"]


def test_postprocess_strips_diacritics_only():
    m = _load_main()
    assert m.postprocess("Nèngdi akèh, é... ya.") == "Nengdi akeh, e... ya."
    assert m.postprocess("Kalau di SMP, ya.") == "Kalau di SMP, ya."
    assert m.postprocess("") == ""


def test_run_survives_transcriber_error(tmp_path: Path):
    m = _load_main()
    data = tmp_path / "data"
    (data / "clips").mkdir(parents=True)
    (data / "clips" / "a.mp3").write_bytes(b"")
    (data / "submission_format.csv").write_text("audio_filename,transcript\na.mp3,\n")
    out = tmp_path / "submission.csv"

    def boom(_):
        raise RuntimeError("decode failed")

    m.run(boom, data, out)
    df = pd.read_csv(out, keep_default_na=False)
    assert df["transcript"].tolist() == [""]
