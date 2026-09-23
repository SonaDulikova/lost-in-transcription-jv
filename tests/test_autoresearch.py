import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lit.autoresearch import (
    DEFAULT_DECODE, Store, decode_cmd, export_cmd, make_row, run_logged, score_convo,
    table_md, train_cmd,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "autoresearch")


def test_ids_are_per_kind_and_zero_padded(store: Store):
    assert store.next_id("decode") == "d001"
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    store.append(make_row("d007", "decode", {}, "h", "ok", 1.0))
    store.append(make_row("t002", "train", {}, "h", "crash", 1.0))
    assert store.next_id("decode") == "d008"
    assert store.next_id("train") == "t003"
    assert store.next_id("postproc") == "p001"


def test_make_row_shape():
    row = make_row("p001", "postproc", {"rules": ["case"]}, "why", "ok", 2.34,
                   metrics={"wer_all": 0.2, "wer_ind": 0.1, "wer_javind": 0.3, "S": 1, "D": 2, "I": 3})
    assert row["id"] == "p001" and row["status"] == "ok" and row["runtime_s"] == 2.3
    assert row["wer_all"] == 0.2 and row["S"] == 1 and row["conclusion"] is None
    assert "error" not in row
    crashed = make_row("t001", "train", {}, None, "crash", 5.0, error="boom")
    assert crashed["wer_all"] is None and crashed["error"] == "boom"


def test_note_only_changes_conclusion(store: Store):
    store.append(make_row("d001", "decode", {"beam": 5}, "h", "ok", 1.0))
    store.append(make_row("d002", "decode", {"beam": 8}, "h", "ok", 1.0))
    store.note("d001", "beam 5 is fine")
    rows = store.rows()
    assert rows[0]["conclusion"] == "beam 5 is fine" and rows[0]["config"] == {"beam": 5}
    assert rows[1]["conclusion"] is None
    with pytest.raises(KeyError):
        store.note("d999", "x")


def test_refusal_reasons(store: Store):
    assert store.refusal("decode") == "no session: run `session --reset`"
    s = store.reset_session(max_hours=12, max_runs=2)
    assert store.refusal("decode") is None
    assert store.refusal("train", gpu_busy=True) == "a scripts/train_lora.py process is running"
    assert store.refusal("decode", gpu_busy=True) is None
    late = datetime.fromisoformat(s["started"]) + timedelta(hours=12, minutes=1)
    assert "12 h elapsed" in store.refusal("decode", now=late)
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    store.append(make_row("d002", "decode", {}, "h", "crash", 1.0))
    assert "2 runs" in store.refusal("decode")
    store.stop.touch()
    assert store.refusal("postproc") == "STOP file present"


def test_reset_session_starts_a_fresh_count(store: Store):
    store.reset_session(max_runs=1)
    store.append(make_row("d001", "decode", {}, "h", "ok", 1.0))
    assert store.refusal("decode") is not None
    store.reset_session(max_runs=1)
    assert store.refusal("decode") is None


def test_table_lists_best_per_kind_and_escapes_pipes():
    rows = [
        make_row("d001", "decode", {"beam": 5}, "a | b", "ok", 300,
                 metrics={"wer_all": 0.19, "wer_ind": 0.13, "wer_javind": 0.2, "S": 1, "D": 1, "I": 1}),
        make_row("d002", "decode", {"beam": 1}, "h", "ok", 200,
                 metrics={"wer_all": 0.18, "wer_ind": 0.13, "wer_javind": 0.2, "S": 1, "D": 1, "I": 1}),
        make_row("t001", "train", {"lr": 1e-4}, "h", "timeout", 2700),
    ]
    md = table_md(rows)
    assert "| decode | d002 | 0.1800 |" in md
    assert "a \\| b" in md
    assert "| t001 |" in md and "timeout" in md
    assert "| train |" not in md.split("## All runs")[0]


@pytest.fixture
def dev_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dev"
    d.mkdir()
    (d / "metadata.tsv").write_text(
        "audio_filename\tlanguage\tconvo_id\n"
        "a.mp3\tind\t2\nb.mp3\tjavind\t2\nc.mp3\tind\t5\n")
    (d / "ground_truth.csv").write_text(
        "audio_filename,transcript\na.mp3,aku iso ya\nb.mp3,neng kene\nc.mp3,ignored\n")
    return d


def test_score_convo_applies_rules_and_filters_convo(dev_dir: Path, tmp_path: Path):
    pred = tmp_path / "p.csv"
    pred.write_text("audio_filename,transcript\nc.mp3,wrong wrong wrong\na.mp3,aku iso ya\nb.mp3,nèng kene. Terima kasih.\n")
    raw = score_convo(pred, [], dev_dir=dev_dir)
    assert raw["wer_ind"] == 0.0 and raw["wer_javind"] > 0
    clean = score_convo(pred, ["diacritics", "tail"], dev_dir=dev_dir)
    assert clean == {"wer_all": 0.0, "wer_ind": 0.0, "wer_javind": 0.0, "S": 0, "D": 0, "I": 0}


def test_score_convo_rejects_missing_rows(dev_dir: Path, tmp_path: Path):
    pred = tmp_path / "p.csv"
    pred.write_text("audio_filename,transcript\na.mp3,aku iso ya\n")
    with pytest.raises(ValueError, match="missing 1"):
        score_convo(pred, [], dev_dir=dev_dir)


def test_run_logged_statuses(tmp_path: Path):
    log = tmp_path / "x.log"
    assert run_logged([sys.executable, "-c", "print('hi')"], log, 10)[0] == "ok"
    assert "hi" in log.read_text()
    status, rc, tail = run_logged([sys.executable, "-c", "import sys; print('bad'); sys.exit(3)"], log, 10)
    assert (status, rc) == ("crash", 3) and "bad" in tail
    status, _, _ = run_logged([sys.executable, "-c", "import time; time.sleep(30)"], log, 0.5)
    assert status == "timeout"


def test_decode_cmd_only_passes_set_options(tmp_path: Path):
    cmd = decode_cmd(Path("runs/ct2/lora_v2"), tmp_path / "d001.csv", DEFAULT_DECODE)
    assert cmd[:3] == ["uv", "run", "scripts/transcribe_dev.py"]
    assert "--no-postprocess" in cmd and "--convo" in cmd and "--patience" not in cmd
    assert "--vad-filter" not in cmd and "--condition-on-previous-text" not in cmd
    cfg = {**DEFAULT_DECODE, "beam": 8, "patience": 1.5, "vad_filter": True}
    cmd = decode_cmd(Path("m"), tmp_path / "d.csv", cfg)
    assert cmd[cmd.index("--beam") + 1] == "8" and cmd[cmd.index("--patience") + 1] == "1.5"
    assert "--vad-filter" in cmd


def test_train_cmd_proxy_overrides_come_last():
    args = ["--train", "data/jember_segments/train.csv", "--epochs", "3", "--lr", "5e-5"]
    cmd = train_cmd("t001", args, Path("runs/t001"), proxy=True, dry=False)
    assert cmd[:3] == ["uv", "run", "scripts/train_lora.py"]
    assert cmd[cmd.index("--val") + 1] == "data/dev_segments/convo2.csv"
    # argparse keeps the last value, so the proxy overrides must follow the agent's flags
    assert cmd[cmd.index("--lr") + 2:] == ["--epochs", "1", "--train-limit", "500", "--val-limit", "40", "--no-wandb", "--run-name", "t001"]
    full = train_cmd("t002", args, Path("runs/t002"), proxy=False, dry=False)
    assert "--no-wandb" not in full and full[full.index("--val-limit") + 1] == "78"
    dry = train_cmd("t003", args, Path("runs/t003"), proxy=True, dry=True)
    assert dry[dry.index("--run-name") + 2:] == ["--epochs", "0.05", "--train-limit", "20", "--val-limit", "4", "--no-wandb"]


def test_export_cmd_uses_base_from_train_args():
    cmd = export_cmd(["--base", "openai/whisper-large-v3", "--train", "x.csv"], Path("runs/t1/adapter"), Path("runs/ct2/t1"))
    assert cmd[cmd.index("--base") + 1] == "openai/whisper-large-v3"
    cmd = export_cmd(["--train", "x.csv"], Path("a"), Path("b"))
    assert cmd[cmd.index("--base") + 1] == "openai/whisper-large-v3-turbo"
