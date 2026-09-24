"""Tests for scripts/experiment.py's CLI wiring: passthrough-flag refusal, the full-run disk
threshold, the full-run session-time check, and the atomic table write."""

import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lit.autoresearch import Store

EXPERIMENT = Path(__file__).resolve().parents[1] / "scripts" / "experiment.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("sub_experiment", EXPERIMENT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sub_experiment"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeUsage:
    def __init__(self, free_gb: float):
        self.free = free_gb * 1e9


@pytest.fixture
def cli(tmp_path, monkeypatch):
    mod = _load_experiment()
    store = Store(tmp_path / "autoresearch")
    monkeypatch.setattr(mod, "Store", lambda: store)
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda path: _FakeUsage(100))
    return mod, store


@pytest.mark.parametrize("flag", ["--out", "--val", "--run-name", "--wandb-project"])
@pytest.mark.parametrize("passthrough", [
    lambda flag: [flag, "somewhere"],       # space-separated form: --out somewhere
    lambda flag: [f"{flag}=somewhere"],     # equals form: --out=somewhere
])
def test_train_refuses_forbidden_passthrough_flags(cli, monkeypatch, capsys, flag, passthrough):
    mod, store = cli
    store.reset_session()
    calls = []
    monkeypatch.setattr(mod, "run_train", lambda *a, **k: calls.append((a, k)))
    rc = mod.main(["train", "--proxy", "--hypothesis", "h", "--", *passthrough(flag)])
    out = capsys.readouterr().out
    assert rc == 2
    assert f"refused: {flag} is set by the runner and cannot be passed through" in out
    assert calls == []
    assert store.rows() == []


def test_train_allows_ordinary_passthrough_flags(cli, monkeypatch):
    mod, store = cli
    store.reset_session()
    calls = []
    monkeypatch.setattr(mod, "run_train", lambda *a, **k: calls.append(a))
    rc = mod.main(["train", "--proxy", "--hypothesis", "h", "--", "--lr", "5e-5"])
    assert rc == 0
    assert calls and calls[0][1] == ["--lr", "5e-5"]


def test_train_full_refused_when_disk_low(cli, monkeypatch, capsys):
    mod, store = cli
    store.reset_session()
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda path: _FakeUsage(30))
    calls = []
    monkeypatch.setattr(mod, "run_train", lambda *a, **k: calls.append(a))
    rc = mod.main(["train", "--full", "--hypothesis", "h"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "refused:" in out and "45" in out
    assert calls == []


def test_train_proxy_only_warns_when_disk_low(cli, monkeypatch, capsys):
    mod, store = cli
    store.reset_session()
    monkeypatch.setattr(mod.shutil, "disk_usage", lambda path: _FakeUsage(10))
    calls = []
    monkeypatch.setattr(mod, "run_train", lambda *a, **k: calls.append(a))
    rc = mod.main(["train", "--proxy", "--hypothesis", "h"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "warning:" in out
    assert calls


def test_train_full_refused_when_session_time_too_short(cli, monkeypatch, capsys):
    mod, store = cli
    store.root.mkdir(parents=True, exist_ok=True)
    started = (datetime.now() - timedelta(hours=8)).isoformat(timespec="seconds")
    store.session.write_text(json.dumps(
        {"started": started, "max_hours": 12.0, "max_runs": 40, "rows_at_start": 0}) + "\n")
    calls = []
    monkeypatch.setattr(mod, "run_train", lambda *a, **k: calls.append(a))
    rc = mod.main(["train", "--full", "--hypothesis", "h"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "refused:" in out and "h left in the session" in out
    assert calls == []


def test_table_write_is_atomic(cli, monkeypatch):
    mod, store = cli
    store.reset_session()
    store.append({"id": "d001", "kind": "decode", "ts": "now", "status": "ok", "config": {},
                  "wer_all": 0.1, "wer_ind": 0.1, "wer_javind": 0.1, "S": 0, "D": 0, "I": 0,
                  "runtime_s": 1.0, "hypothesis": "h", "conclusion": None})
    rc = mod.main(["table"])
    assert rc == 0
    original = store.table.read_text()
    assert "d001" in original

    def boom(*a, **k):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr("lit.autoresearch.os.replace", boom)
    with pytest.raises(OSError):
        mod.main(["table"])
    # the previously-written table must survive a crash before the atomic rename
    assert store.table.read_text() == original
