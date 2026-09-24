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
lora training modes share one `train` leaderboard in that table, so compare
training rows only within the same mode; a low proxy WER is not a full-run
ranking signal.
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
| `train --proxy` | coarse rejection of LoRA/data regressions; 1 epoch on 500 chunks | 45 min hard limit (measured ~9 min) | `train --proxy --hypothesis "..." -- --train data/jember_segments/train.csv --train data/dev_segments/convo5.csv --lr 5e-5 --rank 16` |
| `train --full` | LoRA hyperparameters, data mix, `scripts/train_lora.py` edits at full size | 5 h | 2-4 per night; proxy WER does not rank close candidates (see below) |

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
4. `note` the conclusion. For postproc/decode, keep a code edit only if its row
   beat the current best of that kind; otherwise revert it (`git checkout --
   <file>` on the file you edited is allowed; nothing else in git is). For a
   training edit, revert after a gross proxy regression; otherwise keep it
   only long enough to run the hypothesis-driven full test, then keep/revert
   from that full-run WER. Never keep or revert a close training result from
   proxy ordering alone.
5. Pick the next experiment. Prefer cheap kinds while they still yield
   improvements. A `train --proxy` may reject a clearly bad training change,
   but never rank close candidates: validation measured 0.0036 seed noise and
   ranked v3 ahead of v2 at both 500 and 1000 chunks, opposite the full runs.
6. Spend the remaining night on 2-4 `train --full` runs selected from the
   strongest hypotheses, not from small proxy WER differences. A full run is
   allowed only if more than 5 h remain in the session. If a proxy is used,
   treat only a large regression as evidence to reject; a proxy win never by
   itself justifies or prioritizes a full run.

## Stopping

The runner refuses to start once 12 h have passed since `session --reset`,
after 40 rows, or when `experiments/autoresearch/STOP` exists. When refused,
or when you decide to stop, run `uv run scripts/experiment.py table` and write
`experiments/autoresearch/candidates.md` with three short lists:

- best decode + postproc configuration (id, WER, exact flags);
- full-run candidates, with any proxy result treated as coarse context rather
  than a ranking signal (id, exact flags, why);
- runs that crashed or timed out and what you think caused each.

Then end the session. Do not start another one.
