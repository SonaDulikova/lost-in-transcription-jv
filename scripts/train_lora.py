"""LoRA fine-tuning of Whisper on manifest CSVs (path,text,session,duration)."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from lit.data import strip_diacritics
from lit.wer import corpus_wer

SR = 16000
LANG_NAMES = {"id": "indonesian", "jw": "javanese"}
DEFAULT_WANDB_PROJECT = "lost-in-transcription-jv"


def load_manifests(paths: list[Path], max_seconds: float = 30.0, limit: int | None = None, seed: int = 0) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in paths], ignore_index=True)
    df = df[(df["duration"] <= max_seconds) & (df["text"].str.strip() != "")]
    # Jember writes è/é heavily, dev references almost never do; the scorer keeps diacritics as-is
    df["text"] = df["text"].map(strip_diacritics)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=seed)
    return df.reset_index(drop=True)


def speed_perturb(y: np.ndarray, rng: random.Random) -> np.ndarray:
    factor = rng.choice([1.0, 1.0, 0.9, 1.1])
    if factor == 1.0:
        return y
    return librosa.resample(y, orig_sr=SR, target_sr=int(SR / factor))


def make_transform(processor: WhisperProcessor, augment: bool, seed: int = 0):
    rng = random.Random(seed)

    def transform(batch):
        feats, labels = [], []
        for path, text in zip(batch["path"], batch["text"]):
            y, sr = sf.read(path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            if augment:
                y = speed_perturb(y, rng)
            feats.append(processor.feature_extractor(y, sampling_rate=SR).input_features[0])
            labels.append(processor.tokenizer(text).input_ids)
        return {"input_features": feats, "labels": labels}

    return transform


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (lab[:, 0] == self.decoder_start_token_id).all():
            lab = lab[:, 1:]
        batch["labels"] = lab
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--train", type=Path, action="append", required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--language", default="id", choices=list(LANG_NAMES))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lora-scope", default="all", choices=["all", "decoder", "encoder"],
                    help="which half of the model LoRA adapts; 'all' is the historical default")
    ap.add_argument("--val-limit", type=int, default=150)
    ap.add_argument("--train-limit", type=int, default=None, help="subsample for dry runs")
    ap.add_argument("--sample-seed", type=int, default=0, help="random_state for --train-limit subsampling")
    ap.add_argument("--seed", type=int, default=0,
                    help="trainer seed and augmentation RNG; 0 reproduces every run before 2026-09-24")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true", help="disable W&B logging (use for dry runs)")
    args = ap.parse_args()

    load_dotenv()
    report_to = "none" if args.no_wandb else "wandb"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if "WANDB_API_KEY" not in os.environ or "WANDB_ENTITY" not in os.environ:
            raise RuntimeError("WANDB_API_KEY / WANDB_ENTITY not set; add them to .env or pass --no-wandb")

    processor = WhisperProcessor.from_pretrained(args.base)
    processor.tokenizer.set_prefix_tokens(language=LANG_NAMES[args.language], task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.config.apply_spec_augment = not args.no_augment
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model.generation_config.language = LANG_NAMES[args.language]
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # A plain list matches by module-name suffix (every encoder and decoder block); a string is
    # treated as a regex over the full module name, which is how a single half gets selected.
    targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    if args.lora_scope != "all":
        targets = rf".*\.{args.lora_scope}\..*\.({'|'.join(targets)})"
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      target_modules=targets)
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_df = load_manifests(args.train, limit=args.train_limit, seed=args.sample_seed)
    val_df = load_manifests([args.val], limit=args.val_limit)
    print(f"train {len(train_df)} chunks ({train_df['duration'].sum() / 3600:.2f} h), val {len(val_df)}")
    train_ds = Dataset.from_pandas(train_df[["path", "text"]])
    val_ds = Dataset.from_pandas(val_df[["path", "text"]])
    train_ds.set_transform(make_transform(processor, augment=not args.no_augment, seed=args.seed))
    val_ds.set_transform(make_transform(processor, augment=False))

    def compute_metrics(pred):
        ids = pred.predictions
        ids = np.where(ids == -100, processor.tokenizer.pad_token_id, ids)
        labels = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        refs = processor.batch_decode(labels, skip_special_tokens=True)
        return {"wer": corpus_wer(refs, hyps).wer}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=440,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=4,
        report_to=report_to,
        run_name=args.run_name or args.out.name,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    model.save_pretrained(args.out / "adapter")
    processor.save_pretrained(args.out / "processor")
    print(f"saved adapter to {args.out / 'adapter'}; best eval: {trainer.state.best_metric}")


if __name__ == "__main__":
    main()
