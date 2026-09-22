# lost-in-transcription-jv

Tooling and results for the DrivenData **Lost in Transcription** challenge
(Indonesian and Javanese spontaneous speech): dev-set evaluation that matches the
official scorer, an offline Whisper submission, and LoRA fine-tuning on the
Jember Javanese corpus.

## What the submission contains

`submission_src/` is packed into `submission.zip` and run by the official
runtime image with no internet access.

| file | role |
|---|---|
| `main.py` | reads `submission_format.csv` and `clips/`, transcribes every clip with faster-whisper, writes `submission.csv` in the same row order. Uses GPU float16 when CTranslate2 sees a CUDA device, otherwise CPU int8. Strips diacritics from the output. Logs only counts and timings. |
| `config.json` | decoding settings: language token `id`, beam 5, temperature 0, no conditioning on previous text, no VAD. |
| `model/` | CTranslate2 float16 export of `openai/whisper-large-v3-turbo` with the LoRA adapter `lora_turbo_v2` merged in (about 1.6 GB, not committed). |

Runtime dependencies are pinned to the competition image: `faster-whisper==1.2.1`,
`ctranslate2==4.8.2`, `transformers<5`, Python 3.12.

## Repository layout

```
lit/            normalize.py (verbatim official normalizer), wer.py, data.py, hf_infer.py
scripts/        thin CLIs: dev files, scoring, model fetch, transcription, error report,
                Jember chunking, LoRA training, CT2 export, pack.sh
submission_src/ main.py, config.json, model/ (ignored)
third_party/    score.py copied from the runtime repo, used as the test oracle
tests/          pytest suite (normalizer parity, WER, manifests, main.py I/O)
experiments/    results.md, the full experiment log
```

`data/`, `runs/`, `predictions/`, `wandb/` and all weight files are ignored by git.

## Reproduce

Prerequisites: Linux x86_64, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU for
training (8 GB is enough for turbo LoRA), Docker and `just` for the runtime
check. The runtime repo is expected at `~/repos/lost-in-transcription-runtime`
with the dev data under `data/data/` (override with `LIT_RUNTIME_REPO`).

Data layout expected locally (never committed):

```
data/indonesian_dev/metadata.tsv, clips/          competition dev set
data/jember/<corpus>.tsv, mp3 audio/              Jember corpus (see licence note)
```

1. Environment and tests

   ```bash
   uv sync --group dev
   uv run pytest -q
   ```

2. Dev ground truth and a zero-shot baseline

   ```bash
   uv run scripts/make_dev_files.py
   uv run scripts/fetch_model.py openai/whisper-large-v3-turbo --out submission_src/model
   uv run scripts/transcribe_dev.py --model submission_src/model --language id --beam 5 --out predictions/baseline.csv
   uv run scripts/score.py predictions/baseline.csv
   uv run scripts/error_report.py predictions/baseline.csv --top 30 --out predictions/baseline_errors.md
   ```

3. Training manifests (16 kHz wavs, chunks up to 25 s, validation split by Jember session)

   ```bash
   uv run scripts/prepare_jember.py
   uv run scripts/make_dev_manifest.py
   ```

4. LoRA fine-tuning and export (run 2 recipe: Jember plus dev conversation 5,
   validated on dev conversation 2, which is never trained on)

   ```bash
   uv run scripts/train_lora.py --base openai/whisper-large-v3-turbo \
     --train data/jember_segments/train.csv --train data/dev_segments/convo5.csv \
     --val data/dev_segments/convo2.csv --val-limit 78 \
     --out runs/lora_turbo_v2 --language id --epochs 3
   uv run scripts/export_ct2.py --base openai/whisper-large-v3-turbo \
     --adapter runs/lora_turbo_v2/adapter --out submission_src/model
   uv run scripts/transcribe_dev.py --model submission_src/model --language id --convo 2 --out predictions/c2.csv
   uv run scripts/score.py predictions/c2.csv --convo 2
   ```

   Training logs to Weights & Biases when `WANDB_API_KEY` and `WANDB_ENTITY` are
   in `.env`; pass `--no-wandb` otherwise.

5. Pack and validate offline in the official image (CPU)

   ```bash
   scripts/pack.sh --run --small 10    # 10 clips
   scripts/pack.sh --run               # all 372 dev clips
   ```

   The zip lands in the runtime repo under `submission/submission.zip`.

## Results

Dev-set corpus WER with the official normalizer. Full log with S/D/I counts,
the decoding sweep and error categories: [experiments/results.md](experiments/results.md).

Zero-shot bake-off on all 372 dev clips (GPU float16 unless noted):

| id | model | lang | WER all | WER javind | WER ind | notes |
|---|---|---|---|---|---|---|
| B002 | large-v3-turbo | id | 0.2524 | 0.2575 | 0.1556 | baseline, submitted |
| B003 | large-v3-turbo | id | 0.2544 | 0.2591 | 0.1638 | same zip, Docker CPU int8 |
| B004 | large-v3-turbo | auto | 0.2524 | 0.2575 | 0.1556 | identical to B002 |
| B005 | large-v3-turbo | jw | 0.2857 | 0.2925 | 0.1533 | Javanese token hurts |
| B006 | large-v3 | id | 0.2857 | 0.2905 | 0.1916 | worse and 2x slower |
| B007 | large-v3 | jw | 0.2851 | 0.2898 | 0.1940 | |
| B008 | Qwen3-ASR-1.7B | Indonesian | 0.2537 | 0.2590 | 0.1521 | ties turbo, 3.4x slower |

LoRA runs (rank 32 on large-v3-turbo, lr 1e-4, up to 3 epochs, early stopping),
compared on dev conversation 2 only (78 clips, never in training; beam 5, CT2
float16, diacritics stripped):

| model | training data | WER all | WER ind | WER javind |
|---|---|---|---|---|
| zero-shot turbo | none | 0.2292 | 0.1556 | 0.2527 |
| lora_v1 | Jember only (9.0 h) | 0.2455 | 0.2067 | 0.2579 |
| **lora_v2 (submitted)** | Jember + dev convo 5 (10.9 h) | **0.1834** | **0.1336** | **0.1993** |
| lora_v3 | Jember + convo 5 x2 (12.7 h) | 0.1915 | 0.1370 | 0.2089 |

Jember alone shifts Indonesian speech toward Javanese vocabulary and loses on
convo 2. Adding in-domain dev speech is the lever: 4.6 points over zero-shot
with both languages improving. Beam 5 was kept after a sweep over beam 1/5/8
and `condition_on_previous_text`.

Platform submissions:

| date | type | model | public score |
|---|---|---|---|
| 2026-09-21 | full evaluation | zero-shot turbo (B002) | 0.4697, rank 90 |
| pending | full evaluation | lora_v2 | see `experiments/results.md` |

The public score is far above dev WER for the same zip. The scorer was checked
against `third_party/score.py` and matches, so the hidden test set is harder
than dev; dev deltas rank configurations but absolute levels do not transfer.

## Hardware

All experiments ran on one laptop: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB),
Windows driver 566 under WSL2 Ubuntu. Dev transcription with turbo beam 5 takes
about 9 minutes on the GPU and about 93 minutes on CPU int8 in the runtime
image. A LoRA run takes 3.5 to 4.5 hours. The official image is built on CUDA
13, which this driver cannot expose to containers, so local Docker runs are
CPU-only; the GPU path is only exercised by the platform smoke test.

## Licences

- Code in this repository: Mozilla Public License 2.0, see [LICENSE](LICENSE).
- Base model: `openai/whisper-large-v3-turbo` (MIT).
- The Jember Javanese Spontaneous Speech Corpus is licensed CC BY-NC-SA 4.0.
  It is used here for non-commercial research only and is not redistributed;
  adapters and merged weights trained on it inherit its non-commercial,
  share-alike terms.
- The official scorer in `third_party/score.py` is copied from the
  [runtime repository](https://github.com/drivendataorg/lost-in-transcription-runtime)
  under its own licence.

No audio, filenames or transcripts from the competition data appear in this
repository.
