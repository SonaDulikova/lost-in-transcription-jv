# lost-in-transcription-jv

The DrivenData **Lost in Transcription** challenge: speech recognition
for spontaneous, code-switched Indonesian and Javanese. Finished **29th** on the
private leaderboard (WER 0.2613).

## Approach

- **Base model:** `openai/whisper-large-v3-turbo` with the Indonesian language token.
  Zero-shot it already beat `large-v3` and Qwen3-ASR-1.7B on the dev set.
- **LoRA fine-tuning** (rank 32, all attention and MLP layers) on:
  - TTS Central Javanese read speech from the Mozilla Data Collective (~8 h), which
    matches the dev set's dialect,
  - the competition dev conversations, with clips over 30 s split into two pieces
    at a word boundary.
- **Acoustic augmentation:** MP3 re-encoding, babble noise, reverb, gain and
  band-pass filtering, since the test audio is noisier than the training data.
- **Model soup:** averaged the merged LoRA weights of three seeds.
- **Inference:** faster-whisper (CTranslate2 float16), beam 5, with diacritics and
  hallucinated "terima kasih" endings stripped from the output.

What didn't help: the East Javanese Jember corpus (it made the Indonesian output
worse), extra read speech from OpenSLR SLR35, shrinking the LoRA weights toward
the base model, and rule-based post-processing beyond the two rules above.

## Results

| model | dev WER (held-out conversation) | public leaderboard |
|---|---|---|
| zero-shot turbo | 0.229 | 0.470 |
| LoRA: Jember + dev conversation | 0.183 | 0.278 |
| LoRA: Central Javanese + dev conversation | 0.176 | 0.260 |
| soup of 5 seeds trained on all dev data | – | 0.257 |

The test set scored much worse than dev for every model. Still, the gains on dev
mostly showed up on the leaderboard as well.

## Layout

```
lit/             normalizer, WER, data loading, augmentation, soup, long-clip splitting
scripts/         CLIs: data prep, training, soup, CT2 export, transcription, scoring, packing
submission_src/  the offline submission (main.py, config.json; model weights not included)
tests/           pytest suite
third_party/     the official scorer, used as a test oracle
```

## Usage

```bash
uv sync --group dev
uv run pytest -q

# training data
uv run scripts/make_dev_manifest.py
uv run scripts/prepare_tts.py --tsv data/central_jv/mapping.tsv --audio-dir data/central_jv/audio \
  --file-col audio_filename --text-col sentence

# train, export and score
uv run scripts/train_lora.py --train data/dev_segments/convo5.csv --train data/central_jv/train.csv \
  --val data/dev_segments/convo2.csv --out runs/lora --augment-acoustic --no-wandb
uv run scripts/export_ct2.py --base openai/whisper-large-v3-turbo --adapter runs/lora/adapter --out submission_src/model
uv run scripts/transcribe_dev.py --model submission_src/model --convo 2 --out predictions/c2.csv
uv run scripts/score.py predictions/c2.csv --convo 2
```

The competition data and the training corpora are not included and need to be
downloaded separately into `data/`.

## Licences

Code: MPL 2.0 (see [LICENSE](LICENSE)). Whisper is MIT. TTS Central Javanese is
CC BY-SA 4.0 and Jember is CC BY-NC-SA 4.0; neither is redistributed here. The
official scorer in `third_party/` is under its own licence.
