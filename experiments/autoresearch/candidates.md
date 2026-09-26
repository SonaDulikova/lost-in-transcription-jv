# Autoresearch candidates — night of 2026-09-24/25

Session 20:54 → 03:10. **17 runs, no crashes, no timeouts, nothing reverted.**
Metric everywhere: convo-2 corpus WER, CT2 float16, beam 5, 78 clips,
diacritics stripped. Full log in `results.md` / `results.jsonl`.

## Read this first: the recorded baseline does not reproduce

Retraining v5's exact recipe today (all-scope LoRA r32, convo 5 + Central
Javanese, the flags `runs/overnight3.sh` used) gives **0.1688 ± 0.0026 over
three seeds** — not the **0.1760** in `experiments/results.md`. In-training
epoch-1 val is 0.177/0.177/0.181 today against 0.2166 on 2026-09-22.

Decoding the frozen `runs/ct2/lora_v5` weights still reproduces 0.1760 exactly
(`d001`), so the frozen submission model is unchanged and the drift is in
**training**, not scoring. Something in the environment since 2026-09-22 makes
the same recipe train about 0.007 better.

Everything below is therefore measured against same-night controls. Anything
compared against the 0.1760 on record would show a spurious ~0.007 gain.

## 1. Best decode + postproc configuration

**For the frozen model now in `submission_src/model`** (`runs/ct2/lora_v5`):
`p003`, **0.1744** (ind 0.1336, javind 0.1874), a real −0.0016 for one config
line and no retraining.

    uv run scripts/experiment.py decode --model runs/ct2/lora_v5 --beam 5 --rules diacritics
    uv run scripts/experiment.py postproc --predictions experiments/autoresearch/predictions/d001.csv --rules diacritics,tail

Decoding is unchanged (`language: id`, `beam_size: 5`, no conditioning, no VAD);
the only change is `"postprocess": ["diacritics", "tail"]` in
`submission_src/config.json`. Promoting it needs `lit/postproc.py` copied over
`submission_src/postproc.py` — except that **`lit/postproc.py` was not edited
tonight**, `tail` already existed, so the two files are still byte-identical and
`tests/test_postproc.py` should be green. Only the config line changes.

**Best single number of the night**: `t016` and `t020` tie at **0.1670**, beam 5,
`--rules diacritics`. See the caution about picking the best of nine in §4.

| id | config | WER | verdict |
|---|---|---|---|
| p003 | v5 + diacritics,tail | **0.1744** | keep, −0.0016 |
| p005 | beam-8 decode + diacritics,tail | 0.1746 | independent replication, −0.0023 |
| p006 | decoder-only r32 + diacritics,tail | 0.1699 | still helps, −0.0008 |
| p007 | decoder-only r128 + diacritics,tail | 0.1670 | **no-op**, identical S/D/I |
| p001 | v5 + diacritics | 0.1760 | baseline, matches the record |
| p002 | v5 + diacritics,case | 0.1774 | reject, `case` hurts as on v2 |
| p004 | v5 + diacritics,tail,hai | 0.1744 | `hai` is a no-op |
| d002 | v5 + VAD | 0.1774 | reject: D 78→74 but S 497→508 |
| d003 | v5 beam 8 | 0.1769 | tie at +60% decode time, beam 5 stays |

`tail` (stripping hallucinated `terima kasih` / `sampai jumpa` / `bye` endings)
is the one keeper: it won on two independent v5 decodes and on the decoder-only
r32 model. It is a **no-op on the r128 decoder-only model**, which stopped
emitting those endings. The training fix and the postproc fix correct the same
errors, so do not expect them to add up.

## 2. Full-run candidates

The night's hypothesis was that turbo wastes LoRA capacity: 192 of the 232
targeted modules sit in the 32-layer encoder, which this mix adapts mostly on
8 h of clean read TTS Javanese whose acoustics do not match noisy dev
conversation. `--lora-scope decoder` restricts LoRA to the 4-layer decoder.

**Result: no accuracy difference, large cost difference.**

| arm | seeds | WER | mean | sd | wall/run |
|---|---|---|---|---|---|
| all-scope r32 (v5 recipe) | 0,1,2 | 0.1718 / 0.1676 / 0.1670 | **0.1688** | 0.0026 | 82 min |
| decoder-only r128 | 0,1,2 | 0.1670 / 0.1707 / 0.1715 | **0.1697** | 0.0024 | 25 min |
| decoder-only r32 | 0,1 | 0.1707 / 0.1738 | 0.1723 | 0.0022 | 17–25 min |
| decoder-only r64 | 0 | 0.1693 | — | — | 20 min |

The gap between the two three-seed arms is 0.0009, about a third of one
standard deviation. The hypothesis is **not confirmed on accuracy**. What is
solid is cost: decoder-only r128 trains in a third of the wall time, and r32 in
a fifth, with 4.3 M trainable parameters against 27.9 M.

Ranked for the next night:

1. **Use decoder-only r32 as the screening recipe for data-mix work.** 17–25 min
   a run against 82, at accuracy that three seeds cannot distinguish from the
   full recipe. This is the night's one actionable win: it makes a 3-seed
   comparison of a new data mix cost about one old single run.
2. **Oversample convo 5 in the v5 mix** (`--train data/dev_segments/convo5.csv`
   twice). Untested tonight. Convo 5 is 1.8 h of the 9.8 h mix and in-domain
   conversation was the lever behind v1→v2 (−4.6 pt), while the other 8 h are
   read TTS. Counter-evidence to respect: oversampling convo 5 hurt in the
   Jember-based v2→v3 step. **Run it at three seeds or not at all** — a single
   run cannot resolve anything smaller than about 0.005.
3. **Do not chase LoRA rank.** The r32 → r64 → r128 curve looked monotone at seed
   0 (0.1707 → 0.1693 → 0.1670) and dissolved at seed 1 (r128 0.1707, worse than
   r32 seed 0). It was noise.
4. **Encoder-only LoRA** (`--lora-scope encoder`), diagnostic only. If sparing the
   encoder mattered, encoder-only should be clearly worst. Costs a full 82 min
   and cannot improve the score; run it only to close the mechanism question.

Proxy context, not a ranking signal: `t012` (decoder-only r32, 500 chunks, 1
epoch) scored 0.1783 in 4.5 min. That only showed the edit trains and is not a
gross regression; the full runs went ahead on the hypothesis, not that number.

### Code kept in `scripts/train_lora.py` — please review

Two **additive, opt-in** flags. Both defaults reproduce every pre-2026-09-24 run
byte-for-byte, and no existing behaviour changed (confirmed empirically: `t018`
with all defaults lands in the same family as the historical recipe).

- `--lora-scope {all,decoder,encoder}`, default `all`.
- `--seed N`, default 0, wired to the trainer seed and the augmentation RNG.

I kept them even though the accuracy hypothesis failed, because the defaults are
unchanged, `--seed` is what made the noise floor measurable at all, and
reverting would make tonight's nine full runs irreproducible. If you disagree:
`git checkout -- scripts/train_lora.py`.

`lit/postproc.py` is untouched.

## 3. Crashes and timeouts

**None.** All 17 rows are `status: ok`. Nothing crashed, nothing timed out,
nothing had to be reverted.

Two things that looked wrong and are not, recorded so nobody re-investigates:

- **Identical `eval_wer` to 16 digits across different runs** (0.1710187568959176
  appears in both `t013` and `t016`). Every value is an integer error count over
  the same 2719 reference words on the 65-clip in-training val set, so equal
  values just mean equal error counts. The metric is fine, just coarse.
- **`runs/` grew by about 28 GB** (nine full runs, each keeping an adapter and a
  CT2 export; only proxy runs are cleaned up). 595 GB still free. Old `t0xx`
  directories are safe to delete once you have chosen a model.

## 4. Two findings that should change how results are read here

**Full-run seed noise is 0.0024–0.0027, measured for the first time.** Nine full
runs of configurations that three seeds cannot tell apart span **0.1670 to
0.1738**. Several decisions in `experiments/results.md` rest on margins at or
below that: v4 vs v5 was 0.0037, and the v5-over-v2 freeze was 0.0074 with a
bootstrap CI that already crossed zero. Two or three seeds per arm, or the
paired bootstrap, should gate any future model swap.

This also means **picking the best of nine runs is a biased estimate**. The 0.1670
shared by `t016` and `t020` is the minimum of nine draws from a distribution
centred near 0.169; the honest expectation for either on unseen data is about
0.169–0.170, not 0.167. Both also selected their checkpoint on the same 78 clips
they are scored on, so the dev number is optimistic twice over.

**About a quarter of the remaining errors are unreachable.** Merging
spelling/colloquial variant classes that both conversations use inconsistently
(`gak`/`nggak`/`enggak`, `loh`/`lho`, `'kan`/`kan`, `ya`/`yo`, `kayak`/`kaya`/`kek`,
`tau`/`tahu`, `kui`/`kuwi`, `wis`/`wes`, …) drops v5's 0.1760 to **0.1327**: 154 of
626 errors, **24.6%**, are annotator transcription convention, not recognition
failure. Convo 2 writes `gak` 61 times to `nggak` 29; convo 5, which the model
trains on, writes `nggak` 78 to `gak` 74, and the model duly emits `nggak` 52 to
`gak` 35.

A postproc rule mapping variants toward convo 2's preference would "win" several
points here and generalise to nothing, since the hidden test conversations have
their own transcribers with their own habits. **I deliberately did not add one**,
and any future postproc gain of that shape should be treated as an artefact. The
practical read: the real recognition floor on convo 2 is around 0.13, and the
last few points of measured WER are a coin flip on spelling.
