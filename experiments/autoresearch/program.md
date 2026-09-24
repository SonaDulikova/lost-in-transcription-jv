# Autoresearch program

You are running unattended overnight experiments on a Whisper large-v3-turbo
LoRA for Javanese/Indonesian speech. You run one command, read one number,
log a conclusion, decide the next experiment. Nobody is watching; be careful
and be honest in the log.

## The metric

Corpus WER on dev conversation 2 (78 clips never used in training), CT2
float16, diacritics stripped. Lower is better. Read the current best per kind
from `experiments/autoresearch/results.md` before your first run
(`uv run scripts/experiment.py table` regenerates it). Numbers to beat as of
2026-09-23: decode 0.1760 (lora_v5 = convo 5 + Central Javanese, beam 5),
postproc 0.1760 (diacritics only). v5 is the default `--model` for `decode`.
lora_v5 is also what is currently frozen in `submission_src/model` (second
freeze, 2026-09-23). Full-run history for the data mixes is in
`experiments/results.md` and `predictions/C2_comparison_v4.txt`.

## The one command

    uv run scripts/experiment.py <kind> [flags] --hypothesis "one sentence"

Each call prints its result row as JSON on the last line and appends it to
`experiments/autoresearch/results.jsonl`. `status` is `ok`, `crash` or
`timeout`; a non-ok row is still a data point. After reading the row, record
what you learned:

    uv run scripts/experiment.py note --id d017 --conclusion "one sentence"

Kinds, cheapest first:

| kind | what it varies | budget | example |
|---|---|---|---|
| `postproc` | rule set on an existing CSV | seconds | `postproc --predictions experiments/autoresearch/predictions/d003.csv --rules diacritics,tail,hai` |
| `decode` | CT2 model + decode params | 10 min | `decode --beam 8 --patience 1.5 --rules diacritics` |
| `train --proxy` | LoRA hyperparameters, data mix, `scripts/train_lora.py` edits; 1 epoch on 500 chunks | 45 min (target 25) | `train --proxy --hypothesis "..." -- --train data/jember_segments/train.csv --train data/dev_segments/convo5.csv --lr 5e-5 --rank 16` |
| `train --full` | same, at full size | 5 h | only for a proxy winner (see below) |

Rules available to `--rules`: `diacritics`, `case`, `tail`, `hai`,
`num2words` (see `lit/postproc.py`). Training data manifests:
`data/jember_segments/train.csv` (9.0 h East Javanese),
`data/dev_segments/convo5.csv` (1.8 h in-domain), `data/central_jv/train.csv`
(8.0 h Central Javanese read speech). Flags after `--` go straight to
`scripts/train_lora.py` (`--train`, `--lr`, `--rank`, `--epochs`, `--batch`,
`--accum`, `--no-augment`, `--base`, `--sample-seed`); `--proxy` overrides
`--epochs`, `--train-limit`, `--val-limit` and disables W&B.

## What you may edit

- `lit/postproc.py` — add or change rules (keep existing rule names working).
- `scripts/train_lora.py` — loss, schedule, LoRA targets, augmentation, anything.

Nothing else. In particular never edit or write under `submission_src/`, the
top-level `predictions/` (NOT `experiments/autoresearch/predictions/`, which
the runner writes to itself), `data/`, `runs/ct2/lora_v*`, or
`experiments/results.md`.
Never call the platform, never run git commands that change history or the
remote, never start a second training.

## The loop

1. Write a one-sentence hypothesis for why this change should lower WER.
2. Run exactly one experiment.
3. Read the row. If `crash` or `timeout`, read the tail in `error` and
   `experiments/autoresearch/logs/<id>.log`; revert the edit that caused it
   before trying anything else. Three consecutive non-ok rows: stop and write
   the candidates file.
4. `note` the conclusion. Keep a code edit only if its row beat the current
   best of its kind; otherwise revert it (`git checkout -- <file>` on the
   file you edited is allowed; nothing else in git is).
5. Pick the next experiment. Prefer cheap kinds while they still yield
   improvements; move to `train --proxy` when decode/postproc gains stall.
6. A `train --full` is allowed only when a `--proxy` row beat the best
   proxy row by more than the proxy noise floor: PROXY_NOISE (filled in by
   the human after the validation task), and only if more than 5 h remain in
   the session.

## Stopping

The runner refuses to start once 12 h have passed since `session --reset`,
after 40 rows, or when `experiments/autoresearch/STOP` exists. When refused,
or when you decide to stop, run `uv run scripts/experiment.py table` and write
`experiments/autoresearch/candidates.md` with three short lists:

- best decode + postproc configuration (id, WER, exact flags);
- proxy configurations worth a full run, best first (id, WER, flags, why);
- runs that crashed or timed out and what you think caused each.

Then end the session. Do not start another one.
