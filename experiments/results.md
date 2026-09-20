# Experiment log

One row per experiment. WER is dev-set corpus WER with the official normalizer.

| id | date | model | lang | beam | WER all | WER javind | WER ind | runtime | commit | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| B002 | 2026-09-20 | large-v3-turbo | id | 5 | 0.2524 | 0.2575 | 0.1556 | 510.9s (372 clips, GPU float16) | 2422cff | first turbo baseline, zero-shot |
